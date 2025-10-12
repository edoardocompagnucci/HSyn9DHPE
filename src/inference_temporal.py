"""
Temporal inference for 3D pose estimation with jitter reduction
Uses sliding window approach to maintain temporal smoothness
"""

import os
import cv2
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from models.temporal_model import TemporalPoseModel
from utils.transforms import NormalizerJoints2d
from utils import rotation_utils
from utils.skeleton import SMPL_SKELETON
from utils.detection_smoother import DetectionSmoother
from data.mixed_pose_dataset import coco_wholebody_to_smpl_with_confidence
from mmpose.apis import MMPoseInferencer


def extract_frames_from_video(video_path, output_dir, skip_frames=1, max_frames=None):
    """Extract frames from video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

    frames_dir = Path(output_dir) / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    frame_paths = []
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            frame_filename = frames_dir / f"frame_{frame_count:05d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            frames.append(frame)
            frame_paths.append(str(frame_filename))
            saved_count += 1

            if max_frames and saved_count >= max_frames:
                break

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames")

    return frames, frame_paths, {'width': width, 'height': height, 'fps': fps}


def extract_detections_from_result(result):
    """Extract keypoints from MMPose result"""
    detections = []

    try:
        predictions = result["predictions"][0]

        if isinstance(predictions, list):
            for pred in predictions:
                if "keypoints" in pred and "keypoint_scores" in pred:
                    kpts = np.array(pred["keypoints"], dtype=np.float32)
                    scores = np.array(pred["keypoint_scores"], dtype=np.float32)
                    detections.append((kpts, scores))
        else:
            if "keypoints" in predictions and "keypoint_scores" in predictions:
                kpts = np.array(predictions["keypoints"], dtype=np.float32)
                scores = np.array(predictions["keypoint_scores"], dtype=np.float32)
                detections.append((kpts, scores))

    except (KeyError, IndexError, TypeError) as e:
        print(f"Warning: Could not extract detections: {e}")
        return []

    return detections


def run_2d_detection(frames, pose_detector):
    """Run 2D pose detection on all frames"""
    detections = []

    for frame_idx, frame in enumerate(tqdm(frames, desc="Running 2D detection")):
        try:
            result = next(pose_detector(frame, show=False))
            frame_detections = extract_detections_from_result(result)

            if frame_detections:
                keypoints_coco, scores_coco = frame_detections[0]

                joints_2d_smpl, confidence_weights_smpl = coco_wholebody_to_smpl_with_confidence(
                    keypoints_coco, scores_coco
                )

                detections.append({
                    'frame_idx': frame_idx,
                    'keypoints_coco': keypoints_coco,
                    'scores_coco': scores_coco,
                    'joints_2d_smpl': joints_2d_smpl,
                    'confidence_weights_smpl': confidence_weights_smpl,
                    'detection_failed': False
                })
            else:
                detections.append({
                    'frame_idx': frame_idx,
                    'detection_failed': True
                })
        except Exception as e:
            print(f"Detection failed for frame {frame_idx}: {e}")
            detections.append({
                'frame_idx': frame_idx,
                'detection_failed': True
            })

    return detections


def load_temporal_model(checkpoint_path, device='cuda'):
    """Load temporal pose estimation model"""
    model = TemporalPoseModel(
        num_joints=24,
        dim=384,
        depth=8,
        num_temporal_layers=2,
        temporal_kernel_size=3,
        dropout=0.0
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded temporal model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_pa_mpjpe' in checkpoint:
            print(f"Best PA-MPJPE: {checkpoint['best_val_pa_mpjpe']:.1f}mm")
        if 'best_val_jitter' in checkpoint:
            print(f"Best Jitter: {checkpoint['best_val_jitter']:.1f}mm/s²")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def interpolate_missing_detections(joints_2d_list):
    """
    Fill missing detections using forward/backward interpolation

    Args:
        joints_2d_list: List of [24, 2] arrays or None

    Returns:
        List of [24, 2] arrays with no None values
    """
    filled = joints_2d_list.copy()

    # Forward fill
    last_valid = None
    for i in range(len(filled)):
        if filled[i] is not None:
            last_valid = filled[i]
        elif last_valid is not None:
            filled[i] = last_valid.copy()

    # Backward fill for any remaining None at the start
    last_valid = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is not None:
            last_valid = filled[i]
        elif last_valid is not None:
            filled[i] = last_valid.copy()

    # If still None (all detections failed), use zeros
    for i in range(len(filled)):
        if filled[i] is None:
            filled[i] = np.zeros((24, 2), dtype=np.float32)

    return filled


def run_temporal_3d_pose_lifting(detections, model, normalizer, device='cuda',
                                  image_shape=(1920, 1080), sequence_length=16,
                                  smooth_detections=True):
    """
    Run 3D pose lifting with temporal model using sliding window

    Args:
        detections: List of detection dicts
        model: TemporalPoseModel
        normalizer: NormalizerJoints2d
        device: torch device
        image_shape: (width, height)
        sequence_length: Temporal window size
        smooth_detections: Whether to apply 2D detection smoothing

    Returns:
        List of 3D pose dicts
    """
    poses_3d = []

    # Extract all 2D joints and confidence
    print("Preprocessing detections...")
    joints_2d_list = []
    confidence_list = []

    for detection in detections:
        if detection['detection_failed']:
            joints_2d_list.append(None)
            confidence_list.append(None)
        else:
            joints_2d_list.append(detection['joints_2d_smpl'])
            confidence_list.append(detection['confidence_weights_smpl'])

    # Interpolate missing detections
    joints_2d_filled = interpolate_missing_detections(joints_2d_list)
    confidence_filled = interpolate_missing_detections(confidence_list)

    # Apply detection smoothing to handle outliers
    if smooth_detections:
        print("Smoothing 2D detections to remove outliers...")
        smoother = DetectionSmoother(
            max_displacement_px=100.0,  # Flag jumps > 100px
            min_confidence=0.2,          # Flag low confidence
            use_kalman=False,            # Use simple moving average
            smoothing_window=5           # 5-frame window
        )
        joints_2d_filled, bad_frames = smoother.smooth_detections(
            joints_2d_filled, confidence_filled
        )

        num_bad = sum(bad_frames)
        if num_bad > 0:
            print(f"  Corrected {num_bad}/{len(bad_frames)} frames with bad detections")

    num_frames = len(joints_2d_filled)
    half_window = sequence_length // 2

    print(f"Running temporal 3D pose lifting (window size: {sequence_length})...")

    # Process each frame with sliding window
    for center_idx in tqdm(range(num_frames), desc="Temporal lifting"):
        # Determine window indices (centered on current frame)
        start_idx = center_idx - half_window
        end_idx = start_idx + sequence_length

        # Handle boundary cases with padding
        window_joints = []
        for idx in range(start_idx, end_idx):
            if idx < 0:
                # Pad with first frame
                window_joints.append(joints_2d_filled[0])
            elif idx >= num_frames:
                # Pad with last frame
                window_joints.append(joints_2d_filled[-1])
            else:
                window_joints.append(joints_2d_filled[idx])

        # Stack to [T, J, 2]
        window_np = np.stack(window_joints, axis=0)  # [16, 24, 2]
        window_tensor = torch.from_numpy(window_np).float()

        # Normalize each frame in the window
        normalized_window = []
        for t in range(sequence_length):
            sample = {
                "joints_2d": window_tensor[t],
                "resolution": torch.tensor([image_shape[0], image_shape[1]], dtype=torch.float32)
            }
            normalized_sample = normalizer(sample)
            normalized_window.append(normalized_sample["joints_2d"])

        window_normalized = torch.stack(normalized_window)  # [16, 24, 2]
        window_batch = window_normalized.unsqueeze(0).to(device)  # [1, 16, 24, 2]

        # Forward pass through temporal model
        try:
            with torch.no_grad():
                pos3d_flat, rot6d_flat = model(window_batch)  # [1, 16, J*3], [1, 16, J, 6]

            # Extract center frame from window output
            joints_3d = pos3d_flat.reshape(1, sequence_length, 24, 3)
            joints_3d_center = joints_3d[0, half_window].cpu().numpy()  # [24, 3]

            rotations_6d = rot6d_flat.reshape(1, sequence_length, 24, 6)
            rotations_6d_center = rotations_6d[0, half_window].cpu().numpy()  # [24, 6]

            # Convert 6D rotation to 3x3 matrix
            rotations_6d_torch = torch.from_numpy(rotations_6d_center).float()
            rotations_3x3_torch = rotation_utils.rot_6d_to_matrix(rotations_6d_torch)
            rotations_3x3 = rotations_3x3_torch.cpu().numpy().tolist()

            pose_data = {
                'joints_3d': joints_3d_center,
                'rotations_6d': rotations_6d_center,
                'rotations_3x3': rotations_3x3
            }
            poses_3d.append(pose_data)

        except Exception as e:
            print(f"3D lifting failed for frame {center_idx}: {e}")
            poses_3d.append(None)

    return poses_3d


def visualize_3d_pose(joints_3d, save_path=None, title="3D Pose"):
    """Visualize 3D skeleton"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2],
               c='red', s=50, alpha=0.8)

    for conn in SMPL_SKELETON:
        ax.plot([joints_3d[conn[0], 0], joints_3d[conn[1], 0]],
                [joints_3d[conn[0], 1], joints_3d[conn[1], 1]],
                [joints_3d[conn[0], 2], joints_3d[conn[1], 2]],
                'b-', linewidth=2, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    max_range = np.array([
        joints_3d[:, 0].max() - joints_3d[:, 0].min(),
        joints_3d[:, 1].max() - joints_3d[:, 1].min(),
        joints_3d[:, 2].max() - joints_3d[:, 2].min()
    ]).max() / 2.0

    mid_x = (joints_3d[:, 0].max() + joints_3d[:, 0].min()) * 0.5
    mid_y = (joints_3d[:, 1].max() + joints_3d[:, 1].min()) * 0.5
    mid_z = (joints_3d[:, 2].max() + joints_3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Temporal 3D pose estimation from video')
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('output_dir', type=str, help='Output directory for results')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to temporal model checkpoint')
    parser.add_argument('--sequence_length', type=int, default=16,
                        help='Temporal window size (default: 16)')
    parser.add_argument('--skip_frames', type=int, default=1,
                        help='Process every Nth frame (default: 1)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames to process (default: all)')
    parser.add_argument('--visualize_3d', action='store_true',
                        help='Generate 3D pose visualizations')
    parser.add_argument('--smooth_detections', action='store_true',
                        help='Apply 2D detection smoothing to remove outliers (recommended)')

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video not found: {args.video_path}")
        return

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("TEMPORAL 3D POSE ESTIMATION")
    print("="*60)

    # Step 1: Extract frames
    print("\n[Step 1/3] Extracting frames from video...")
    frames, frame_paths, video_info = extract_frames_from_video(
        args.video_path, args.output_dir, args.skip_frames, args.max_frames
    )

    # Step 2: 2D detection
    print("\n[Step 2/3] Running 2D pose detection...")
    pose_detector = MMPoseInferencer(
        pose2d="rtmpose-l_8xb32-270e_coco-wholebody-384x288",
        device=str(device)
    )

    detections = run_2d_detection(frames, pose_detector)

    successful = sum(1 for d in detections if not d['detection_failed'])
    print(f"Successful detections: {successful}/{len(detections)}")

    # Step 3: Temporal 3D lifting
    print("\n[Step 3/3] Running temporal 3D pose lifting...")

    model = load_temporal_model(args.checkpoint, str(device))
    normalizer = NormalizerJoints2d()

    poses_3d = run_temporal_3d_pose_lifting(
        detections, model, normalizer, str(device),
        image_shape=(video_info['width'], video_info['height']),
        sequence_length=args.sequence_length,
        smooth_detections=args.smooth_detections
    )

    successful_3d = sum(1 for p in poses_3d if p is not None)
    print(f"Successful 3D poses: {successful_3d}/{len(poses_3d)}")

    # Optional: Visualize 3D poses
    if args.visualize_3d:
        vis_3d_dir = output_path / 'visualizations_3d'
        vis_3d_dir.mkdir(exist_ok=True)
        print("\nGenerating 3D visualizations...")
        for i, pose_data in enumerate(tqdm(poses_3d, desc="Saving 3D visualizations")):
            if pose_data is not None:
                vis_path = vis_3d_dir / f"frame_{i:05d}_3d.png"
                visualize_3d_pose(
                    pose_data['joints_3d'],
                    save_path=str(vis_path),
                    title=f"Frame {i}"
                )

    # Save results
    print("\nSaving results...")

    # Combined results JSON
    results = []
    for i, (detection, pose_data) in enumerate(zip(detections, poses_3d)):
        result = detection.copy()
        if pose_data is not None:
            result['joints_3d'] = pose_data['joints_3d'].tolist()
            result['rotations_6d'] = pose_data['rotations_6d'].tolist()
            result['rotations_3x3'] = pose_data['rotations_3x3']
        else:
            result['joints_3d'] = None
            result['rotations_6d'] = None
            result['rotations_3x3'] = None
        results.append(result)

    json_path = output_path / 'results_temporal.json'
    with open(json_path, 'w') as f:
        json_results = []
        for r in results:
            json_r = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    json_r[k] = v.tolist()
                else:
                    json_r[k] = v
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)

    print(f"Results saved to: {json_path}")

    # Houdini frame files
    houdini_dir = output_path / 'houdini_frames'
    houdini_dir.mkdir(exist_ok=True)

    print("Saving Houdini frame files...")
    for i, (detection, pose_data) in enumerate(zip(detections, poses_3d)):
        if pose_data is not None:
            joints_3d_list = pose_data['joints_3d'].tolist() if isinstance(pose_data['joints_3d'], np.ndarray) else pose_data['joints_3d']
            rotations_6d_list = pose_data['rotations_6d'].tolist() if isinstance(pose_data['rotations_6d'], np.ndarray) else pose_data['rotations_6d']
            confidence_list = detection['confidence_weights_smpl'] if isinstance(detection['confidence_weights_smpl'], list) else detection['confidence_weights_smpl'].tolist()

            houdini_data = {
                "frame": i,
                "pose_3d_bone_corrected": joints_3d_list,
                "rotations_3x3": pose_data['rotations_3x3'],
                "rotations_6d": rotations_6d_list,
                "confidence_weights": confidence_list
            }

            frame_json_path = houdini_dir / f'frame_{i:05d}.json'
            with open(frame_json_path, 'w') as f:
                json.dump(houdini_data, f, indent=2)

    print(f"Houdini frame files saved to: {houdini_dir}")

    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Video: {Path(args.video_path).name}")
    print(f"Resolution: {video_info['width']}x{video_info['height']}")
    print(f"FPS: {video_info['fps']:.2f}")
    print(f"Total frames: {len(frames)}")
    print(f"Successful 2D detections: {successful}/{len(detections)}")
    print(f"Successful 3D poses: {successful_3d}/{len(poses_3d)}")
    print(f"\nResults saved to: {output_path}")
    print(f"  - results_temporal.json: Complete pose data")
    print(f"  - houdini_frames/: Per-frame JSON files for Houdini")
    if args.visualize_3d:
        print(f"  - visualizations_3d/: 3D pose visualizations")
    print("\nNote: This output uses temporal smoothing for reduced jitter!")


if __name__ == "__main__":
    main()
