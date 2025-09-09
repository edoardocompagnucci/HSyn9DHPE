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

from models.graphformer import GraphFormerPose
from utils.transforms import NormalizerJoints2d
from utils import rotation_utils
from utils.skeleton import SMPL_SKELETON
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
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model

def run_3d_pose_lifting(detections, model, normalizer, device='cuda', image_shape=(1920, 1080)):
    poses_3d = []
    
    for detection in tqdm(detections, desc="3D pose lifting"):
        if detection['detection_failed']:
            poses_3d.append(None)
            continue
        
        try:
            joints_2d_smpl = np.array(detection['joints_2d_smpl'])
            
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
            print(f"3D lifting failed for frame {detection['frame_idx']}: {e}")
            poses_3d.append(None)
    
    return poses_3d

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
    parser = argparse.ArgumentParser(description='Video to 3D pose estimation')
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('output_dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    checkpoint = 'checkpoints/train_20250905_230529_graphformer_detected_pa_mpjpe/best_model.pth'
    skip_frames = 1
    max_frames = None
    visualize_3d = False
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video not found: {args.video_path}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    vis_3d_dir = output_path / 'visualizations_3d'
    
    print("VIDEO TO 3D POSE ESTIMATION")
    
    print("\n[Step 1/3] Extracting frames from video...")
    frames, frame_paths, video_info = extract_frames_from_video(
        args.video_path, args.output_dir, skip_frames, max_frames
    )
    
    print("\n[Step 2/3] Running 2D pose detection...")
    pose_detector = MMPoseInferencer(
        pose2d="rtmpose-l_8xb32-270e_coco-wholebody-384x288",
        device=str(device)
    )
    
    detections = run_2d_detection(frames, pose_detector)
    
    successful = sum(1 for d in detections if not d['detection_failed'])
    print(f"Successful detections: {successful}/{len(detections)}")
    
    print("\n[Step 3/3] Running 3D pose lifting...")
    
    model = load_graphformer_model(checkpoint, str(device))
    normalizer = NormalizerJoints2d()
    
    poses_3d = run_3d_pose_lifting(
        detections, model, normalizer, str(device),
        image_shape=(video_info['width'], video_info['height'])
    )
    
    successful_3d = sum(1 for p in poses_3d if p is not None)
    print(f"Successful 3D poses: {successful_3d}/{len(poses_3d)}")
    
    if visualize_3d:
        vis_3d_dir.mkdir(exist_ok=True)
        print("\nVisualizing 3D poses...")
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
    
    json_path = output_path / 'results.json'
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
    
    print("\nPROCESSING COMPLETE")
    print(f"Video: {Path(args.video_path).name}")
    print(f"Resolution: {video_info['width']}x{video_info['height']}")
    print(f"Total frames: {len(frames)}")
    print(f"Successful 2D detections: {successful}/{len(detections)}")
    print(f"Successful 3D poses: {successful_3d}/{len(poses_3d)}")
    print(f"\nResults saved to: {output_path}")
    print(f"  - results.json: Complete pose data")
    print(f"  - houdini_frames/: Per-frame JSON files for Houdini")

if __name__ == "__main__":
    main()