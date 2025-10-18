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
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from models.graphformer import GraphFormerPose
from utils.transforms import NormalizerJoints2d
from utils import rotation_utils
from utils.skeleton import SMPL_SKELETON
from utils.detection_smoother import DetectionSmoother
from data.mixed_pose_dataset import coco_wholebody_to_smpl_with_confidence
from mmpose.apis import MMPoseInferencer


def extract_frames_from_video(video_path, output_dir, skip_frames=1, max_frames=None):
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


def load_graphformer_model(checkpoint_path, device='cuda'):
    model = GraphFormerPose(
        num_joints=24,
        dim=384,
        depth=8,
        heads=12,
        ffn_dim=1536,
        dropout=0.0
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best PA-MPJPE: {checkpoint.get('val_pa_mpjpe', 'N/A'):.1f}mm")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_pa_mpjpe' in checkpoint:
            pa_mpjpe_mm = checkpoint['val_pa_mpjpe'] * 1000
            print(f"Best PA-MPJPE: {pa_mpjpe_mm:.1f}mm")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def interpolate_missing_detections(joints_2d_list):
    filled = joints_2d_list.copy()

    last_valid = None
    for i in range(len(filled)):
        if filled[i] is not None:
            last_valid = filled[i]
        elif last_valid is not None:
            filled[i] = last_valid.copy()

    last_valid = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is not None:
            last_valid = filled[i]
        elif last_valid is not None:
            filled[i] = last_valid.copy()

    for i in range(len(filled)):
        if filled[i] is None:
            filled[i] = np.zeros((24, 2), dtype=np.float32)

    return filled


def run_3d_pose_lifting(detections, model, normalizer, device='cuda',
                        image_shape=(1920, 1080), smooth_detections=False):
    joints_2d_list = []
    confidence_list = []

    for detection in detections:
        if detection['detection_failed']:
            joints_2d_list.append(None)
            confidence_list.append(None)
        else:
            joints_2d_list.append(detection['joints_2d_smpl'])
            confidence_list.append(detection['confidence_weights_smpl'])

    joints_2d_filled = interpolate_missing_detections(joints_2d_list)
    confidence_filled = interpolate_missing_detections(confidence_list)

    if smooth_detections:
        print("Smoothing 2D detections to remove outliers...")
        smoother = DetectionSmoother(
            max_displacement_px=100.0,
            min_confidence=0.2,
            use_kalman=False,
            smoothing_window=5
        )
        joints_2d_filled, bad_frames = smoother.smooth_detections(
            joints_2d_filled, confidence_filled
        )
        num_bad = sum(bad_frames)
        if num_bad > 0:
            print(f"  Corrected {num_bad}/{len(bad_frames)} frames with bad detections")

    poses_3d = []

    for detection_idx, joints_2d_smpl in enumerate(tqdm(joints_2d_filled, desc="3D pose lifting")):
        try:
            if joints_2d_smpl.shape != (24, 2):
                print(f"Warning: Unexpected joints shape {joints_2d_smpl.shape}, expected (24, 2)")
                if joints_2d_smpl.size == 48:
                    joints_2d_smpl = joints_2d_smpl.reshape(24, 2)
                else:
                    poses_3d.append(None)
                    continue

            joints_2d_torch = torch.from_numpy(joints_2d_smpl).float()

            sample = {
                "joints_2d": joints_2d_torch,
                "resolution": torch.tensor([image_shape[0], image_shape[1]])
            }
            normalized_sample = normalizer(sample)
            joints_2d_normalized = normalized_sample["joints_2d"]
            joints_2d_tensor = joints_2d_normalized.unsqueeze(0).to(device)

            with torch.no_grad():
                pos3d_flat, rot6d_flat = model(joints_2d_tensor)

            joints_3d = pos3d_flat.reshape(-1, 24, 3).squeeze(0).cpu().numpy()
            rotations_6d = rot6d_flat.reshape(-1, 24, 6).squeeze(0).cpu().numpy()

            rotations_6d_torch = torch.from_numpy(rotations_6d).float()
            rotations_3x3_torch = rotation_utils.rot_6d_to_matrix(rotations_6d_torch)
            rotations_3x3 = rotations_3x3_torch.cpu().numpy().tolist()

            pose_data = {
                'joints_3d': joints_3d,
                'rotations_6d': rotations_6d,
                'rotations_3x3': rotations_3x3
            }
            poses_3d.append(pose_data)

        except Exception as e:
            print(f"3D lifting failed for frame {detection_idx}: {e}")
            poses_3d.append(None)

    return poses_3d


def smooth_3d_poses(poses_3d, window_length=9, polyorder=3):
    print(f"Applying 3D smoothing (Savitzky-Golay window={window_length}, poly={polyorder})...")

    valid_indices = [i for i, p in enumerate(poses_3d) if p is not None]
    if len(valid_indices) < window_length:
        print(f"Warning: Not enough valid frames ({len(valid_indices)}) for smoothing window ({window_length})")
        return poses_3d

    positions_array = np.array([poses_3d[i]['joints_3d'] for i in valid_indices])
    rotations_6d_array = np.array([poses_3d[i]['rotations_6d'] for i in valid_indices])

    T, J, _ = positions_array.shape

    smoothed_positions = np.zeros_like(positions_array)
    for joint_idx in range(J):
        for coord_idx in range(3):
            smoothed_positions[:, joint_idx, coord_idx] = savgol_filter(
                positions_array[:, joint_idx, coord_idx],
                window_length=window_length,
                polyorder=polyorder,
                mode='nearest'
            )

    smoothed_rotations_6d = np.zeros_like(rotations_6d_array)

    for joint_idx in range(J):
        rot_6d_joint = torch.from_numpy(rotations_6d_array[:, joint_idx, :]).float()
        rot_matrices = rotation_utils.rot_6d_to_matrix(rot_6d_joint).numpy()

        rotations = R.from_matrix(rot_matrices)

        smoothed_rots = []
        half_window = window_length // 2

        for t in range(T):
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)

            window_rots = rotations[start:end]

            if len(window_rots) == 1:
                smoothed_rots.append(window_rots[0])
            else:
                middle_idx = len(window_rots) // 2
                if middle_idx > 0 and middle_idx < len(window_rots) - 1:
                    prev_rot = window_rots[middle_idx - 1]
                    curr_rot = window_rots[middle_idx]
                    next_rot = window_rots[middle_idx + 1]

                    times = [0, 1, 2]
                    key_rots = R.from_quat([prev_rot.as_quat(), curr_rot.as_quat(), next_rot.as_quat()])
                    slerp = Slerp(times, key_rots)
                    smoothed_rot = slerp([1.0])[0]
                else:
                    smoothed_rot = window_rots[middle_idx]

                smoothed_rots.append(smoothed_rot)

        smoothed_rot_matrices = R.from_quat([r.as_quat() for r in smoothed_rots]).as_matrix()
        smoothed_rot_matrices_torch = torch.from_numpy(smoothed_rot_matrices).float()
        smoothed_6d = rotation_utils.rot_matrix_to_6d(smoothed_rot_matrices_torch).numpy()

        smoothed_rotations_6d[:, joint_idx, :] = smoothed_6d

    smoothed_poses = []
    valid_idx = 0

    for i in range(len(poses_3d)):
        if i in valid_indices:
            rot_6d_torch = torch.from_numpy(smoothed_rotations_6d[valid_idx]).float()
            rot_3x3_torch = rotation_utils.rot_6d_to_matrix(rot_6d_torch)
            rot_3x3 = rot_3x3_torch.numpy().tolist()

            smoothed_pose = {
                'joints_3d': smoothed_positions[valid_idx],
                'rotations_6d': smoothed_rotations_6d[valid_idx],
                'rotations_3x3': rot_3x3
            }
            smoothed_poses.append(smoothed_pose)
            valid_idx += 1
        else:
            smoothed_poses.append(None)

    print(f"Smoothed {len(valid_indices)} frames")
    return smoothed_poses


def visualize_3d_pose(joints_3d, save_path=None, title="3D Pose"):
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
    parser = argparse.ArgumentParser(description='Per-frame 3D pose estimation with optional smoothing')
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('output_dir', type=str, help='Output directory for results')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to per-frame model checkpoint (e.g., 57mm model)')
    parser.add_argument('--skip_frames', type=int, default=1,
                        help='Process every Nth frame (default: 1)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames to process (default: all)')
    parser.add_argument('--visualize_3d', action='store_true',
                        help='Generate 3D pose visualizations')
    parser.add_argument('--smooth_detections', action='store_true',
                        help='Apply 2D detection smoothing to remove outliers')
    parser.add_argument('--smooth_3d', action='store_true',
                        help='Apply 3D pose smoothing (Savitzky-Golay for positions, SLERP for rotations)')
    parser.add_argument('--smooth_window', type=int, default=9,
                        help='Smoothing window size (must be odd, default: 9)')

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

    print("\nExtracting frames from video...")
    frames, frame_paths, video_info = extract_frames_from_video(
        args.video_path, args.output_dir, args.skip_frames, args.max_frames
    )

    print("\nRunning 2D pose detection...")
    pose_detector = MMPoseInferencer(
        pose2d="rtmpose-l_8xb32-270e_coco-wholebody-384x288",
        device=str(device)
    )

    detections = run_2d_detection(frames, pose_detector)

    successful = sum(1 for d in detections if not d['detection_failed'])
    print(f"Successful detections: {successful}/{len(detections)}")

    print("\nRunning per-frame 3D pose lifting...")

    model = load_graphformer_model(args.checkpoint, str(device))
    normalizer = NormalizerJoints2d()

    poses_3d = run_3d_pose_lifting(
        detections, model, normalizer, str(device),
        image_shape=(video_info['width'], video_info['height']),
        smooth_detections=args.smooth_detections
    )

    successful_3d = sum(1 for p in poses_3d if p is not None)
    print(f"Successful 3D poses: {successful_3d}/{len(poses_3d)}")

    if args.smooth_3d:
        print("\nApplying 3D pose smoothing...")
        poses_3d = smooth_3d_poses(poses_3d, window_length=args.smooth_window, polyorder=3)
    else:
        print("\nSkipping 3D smoothing (use --smooth_3d to enable)")

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

    print("\nSaving results...")

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

    json_path = output_path / 'results_frame.json'
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

    print("\nProcessing complete")
    print(f"Video: {Path(args.video_path).name}")
    print(f"Resolution: {video_info['width']}x{video_info['height']}")
    print(f"FPS: {video_info['fps']:.2f}")
    print(f"Total frames: {len(frames)}")
    print(f"Successful 2D detections: {successful}/{len(detections)}")
    print(f"Successful 3D poses: {successful_3d}/{len(poses_3d)}")
    print(f"\nResults saved to: {output_path}")
    print(f"  - results_frame.json: Complete pose data")
    print(f"  - houdini_frames/: Per-frame JSON files for Houdini")
    if args.visualize_3d:
        print(f"  - visualizations_3d/: 3D pose visualizations")
    if args.smooth_3d:
        print("\n3D smoothing applied (Savitzky-Golay + SLERP)")
    else:
        print("\nUse --smooth_3d flag to enable post-processing smoothing")


if __name__ == "__main__":
    main()
