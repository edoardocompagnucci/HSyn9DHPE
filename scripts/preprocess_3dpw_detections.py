import os
import pickle
import json
import numpy as np
import cv2
from tqdm import tqdm
import torch
from mmpose.apis import MMPoseInferencer
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

THREEDPW_ROOT = os.path.join(PROJECT_ROOT, "data", "3DPW")
SEQUENCE_FILES_DIR = os.path.join(THREEDPW_ROOT, "sequenceFiles")
IMAGE_FILES_DIR = os.path.join(THREEDPW_ROOT, "imageFiles")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "3DPW_processed")
DETECTIONS_DIR = os.path.join(OUTPUT_DIR, "detections")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

REST_POSE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "meta", "rest_pose_data.npy")

SAVE_VISUALIZATIONS = True
VIS_EVERY_N_FRAMES = 10

def load_rest_pose_data():
    if not os.path.exists(REST_POSE_DATA_PATH):
        raise FileNotFoundError(f"Rest pose data not found: {REST_POSE_DATA_PATH}")
    
    data = np.load(REST_POSE_DATA_PATH, allow_pickle=True).item()
    print(f"Loaded rest pose data: {data['num_joints']} joints")
    return data

def visualize_coco_wholebody_detections(img_original, detections_original, seq_name, frame_idx):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    
    keypoint_groups = {
        'body': {'indices': list(range(17)), 'color': 'red', 'size': 40, 'label': 'Body'},
        'foot': {'indices': list(range(17, 23)), 'color': 'orange', 'size': 35, 'label': 'Feet'},
        'face': {'indices': list(range(23, 91)), 'color': 'yellow', 'size': 15, 'label': 'Face'},
        'left_hand': {'indices': list(range(91, 112)), 'color': 'cyan', 'size': 25, 'label': 'Left Hand'},
        'right_hand': {'indices': list(range(112, 133)), 'color': 'magenta', 'size': 25, 'label': 'Right Hand'}
    }
    
    for person_idx, (kpts, scores) in enumerate(detections_original):
        for group_name, group_info in keypoint_groups.items():
            indices = group_info['indices']
            group_kpts = kpts[indices]
            group_scores = scores[indices]
            
            valid_mask = group_scores > 0.3
            valid_kpts = group_kpts[valid_mask]
            
            if len(valid_kpts) > 0:
                ax.scatter(valid_kpts[:, 0], valid_kpts[:, 1], 
                          c=group_info['color'], 
                          s=group_info['size'],
                          alpha=0.8,
                          label=f"Person {person_idx} - {group_info['label']}" if person_idx == 0 else "")
        
        body_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        
        for conn in body_connections:
            if scores[conn[0]] > 0.3 and scores[conn[1]] > 0.3:
                ax.plot([kpts[conn[0], 0], kpts[conn[1], 0]], 
                       [kpts[conn[0], 1], kpts[conn[1], 1]], 
                       'lime', linewidth=2, alpha=0.6)
        
        foot_connections = [
            (15, 17), (15, 18), (15, 19),
            (16, 20), (16, 21), (16, 22)
        ]
        
        for conn in foot_connections:
            if scores[conn[0]] > 0.3 and scores[conn[1]] > 0.3:
                ax.plot([kpts[conn[0], 0], kpts[conn[1], 0]], 
                       [kpts[conn[0], 1], kpts[conn[1], 1]], 
                       'orange', linewidth=2, alpha=0.6)
        
        body_kpts = kpts[:17]
        body_scores = scores[:17]
        valid_body = body_kpts[body_scores > 0.3]
        if len(valid_body) > 0:
            x_min, y_min = valid_body.min(axis=0)
            x_max, y_max = valid_body.max(axis=0)
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                   linewidth=2, edgecolor='white', facecolor='none', 
                                   linestyle='--', alpha=0.8)
            ax.add_patch(rect)
            
            ax.text(x_min, y_min-5, f'Person {person_idx}', 
                   color='white', fontsize=12, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
    
    stats_text = f"Frame {frame_idx}\n"
    stats_text += f"Resolution: {img_original.shape[1]}x{img_original.shape[0]}\n"
    for person_idx, (kpts, scores) in enumerate(detections_original):
        valid_body = np.sum(scores[:17] > 0.3)
        valid_feet = np.sum(scores[17:23] > 0.3)
        valid_hands = np.sum(scores[91:133] > 0.3)
        stats_text += f"Person {person_idx}: Body={valid_body}/17, Feet={valid_feet}/6, Hands={valid_hands}/42\n"
    
    ax.text(10, 30, stats_text, color='white', fontsize=10, weight='bold',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7))
    
    ax.set_title(f'{seq_name} - Frame {frame_idx} (Original Resolution)', fontsize=14, weight='bold')
    ax.set_xlim(0, img_original.shape[1])
    ax.set_ylim(img_original.shape[0], 0)
    ax.axis('off')
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(os.path.join(VISUALIZATIONS_DIR, seq_name), exist_ok=True)
    save_path = os.path.join(VISUALIZATIONS_DIR, seq_name, f"frame_{frame_idx:05d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

def axis_angle_to_rotation_matrix(axis_angle):
    angle = np.linalg.norm(axis_angle)
    
    if angle < 1e-8:
        return np.eye(3)
    
    axis = axis_angle / angle
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
    ])

def forward_kinematics(local_rotations, parents):
    rot_mats = np.zeros_like(local_rotations)
    
    for joint_id in range(len(parents)):
        parent_id = parents[joint_id]
        
        if parent_id == -1:
            rot_mats[joint_id] = local_rotations[joint_id]
        else:
            rot_mats[joint_id] = rot_mats[parent_id] @ local_rotations[joint_id]
    
    return rot_mats

def rot_matrix_to_6d(rot_matrices):
    return np.concatenate([rot_matrices[..., :, 0], rot_matrices[..., :, 1]], axis=-1)

def process_rotations_houdini_pipeline(pose_params, rest_pose_data):
    axis_angles = pose_params.reshape(24, 3)
    smpl_local = np.zeros((24, 3, 3))
    for i in range(24):
        smpl_local[i] = axis_angle_to_rotation_matrix(axis_angles[i])
    
    rest_transforms = rest_pose_data['rest_transforms']
    final_local = np.einsum('ijk,ikl->ijl', smpl_local, rest_transforms)
    
    parents = rest_pose_data['smpl_parents']
    rot_mats = forward_kinematics(final_local, parents)
    
    rot_mats = rot_mats.transpose(0, 2, 1)
    
    rot_6d = rot_matrix_to_6d(rot_mats)
    
    return rot_mats, rot_6d

def extract_camera_parameters(seq_data, frame_idx):
    K = seq_data['cam_intrinsics']
    cam_pose = seq_data['cam_poses'][frame_idx]
    R = cam_pose[:3, :3]
    t = cam_pose[:3, 3]
    return K, R, t

def project_3d_to_2d(joints_3d_world, cam_intrinsics, cam_pose):
    joints_3d_cam = (cam_pose[:3, :3] @ joints_3d_world.T + cam_pose[:3, 3:4]).T
    projected = (cam_intrinsics @ joints_3d_cam.T).T
    joints_2d = projected[:, :2] / projected[:, 2:3]
    return joints_2d

def smpl_to_coco_body_subset(smpl_joints_2d):
    coco_body = np.zeros((17, 2), dtype=np.float32)
    
    coco_body[0] = smpl_joints_2d[15]  # nose = Head
    coco_body[5] = smpl_joints_2d[17]  # left_shoulder = R_Shoulder (SMPL)
    coco_body[6] = smpl_joints_2d[16]  # right_shoulder = L_Shoulder (SMPL)
    coco_body[7] = smpl_joints_2d[19]  # left_elbow = R_Elbow
    coco_body[8] = smpl_joints_2d[18]  # right_elbow = L_Elbow
    coco_body[9] = smpl_joints_2d[21]  # left_wrist = R_Wrist
    coco_body[10] = smpl_joints_2d[20]  # right_wrist = L_Wrist
    coco_body[11] = smpl_joints_2d[2]  # left_hip = R_Hip
    coco_body[12] = smpl_joints_2d[1]  # right_hip = L_Hip
    coco_body[13] = smpl_joints_2d[5]  # left_knee = R_Knee
    coco_body[14] = smpl_joints_2d[4]  # right_knee = L_Knee
    coco_body[15] = smpl_joints_2d[8]  # left_ankle = R_Ankle
    coco_body[16] = smpl_joints_2d[7]  # right_ankle = L_Ankle
    
    coco_body[1:5] = smpl_joints_2d[15]  # eyes and ears = head
    
    return coco_body

def match_detections_to_actors(detections, gt_joints_2d_list, threshold=250.0):
    if not detections or not gt_joints_2d_list:
        return [None] * len(gt_joints_2d_list)
    
    matches = [None] * len(gt_joints_2d_list)
    used_detections = set()
    
    for actor_idx, gt_joints in enumerate(gt_joints_2d_list):
        best_detection_idx = None
        best_distance = float('inf')
        
        for det_idx, detection in enumerate(detections):
            if det_idx in used_detections:
                continue
                
            det_body = detection[:17]
            distances = np.linalg.norm(det_body - gt_joints, axis=1)
            avg_distance = np.mean(distances)
            
            if avg_distance < best_distance and avg_distance < threshold:
                best_distance = avg_distance
                best_detection_idx = det_idx
        
        if best_detection_idx is not None:
            matches[actor_idx] = best_detection_idx
            used_detections.add(best_detection_idx)
    
    return matches

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
        print(f"Warning: Could not extract detections from result: {e}")
        return []
    
    return detections

def process_sequence(seq_name, split, pose_detector, rest_pose_data):
    print(f"\nProcessing sequence: {seq_name} (split: {split})")
    
    seq_file = os.path.join(SEQUENCE_FILES_DIR, split, f"{seq_name}.pkl")
    if not os.path.exists(seq_file):
        print(f"Sequence file not found: {seq_file}")
        return
    
    with open(seq_file, 'rb') as f:
        seq_data = pickle.load(f, encoding='latin1')
    
    num_actors = len(seq_data['poses'])
    num_frames = len(seq_data['cam_poses'])
    
    print(f"  Actors: {num_actors}, Frames: {num_frames}")
    
    sequence_detections = {
        'sequence_name': seq_name,
        'split': split,
        'num_actors': num_actors,
        'num_frames': num_frames,
        'detections': {},
        'metadata': {
            'detection_method': 'rtmpose-l_8xb32-270e_coco-wholebody-384x288',
            'image_preprocessing': 'NONE_ORIGINAL_RESOLUTION',
            'keypoint_format': 'coco_wholebody_133_joints_in_original_pixel_space',
            'keypoint_detection': 'detected_on_original_resolution',
            'rotation_format': '6d_representation_24_joints_EXACT_SAME_AS_HOUDINI_PIPELINE',
            'rotation_processing': 'axis_angle_to_rotation_matrix_then_FK_then_transpose_then_6d',
            '3d_positions_format': 'root_centered_24_joints_same_as_synthetic_dataset',
            'camera_format': 'K_R_t_resolution_same_as_synthetic_dataset',
            'resolution_format': '[width, height]',
            'root_translation_format': 'original_root_position_before_centering',
            'forward_kinematics_applied': True,
            'rest_pose_transformation_applied': True,
            'transpose_applied_after_FK': True,
            'houdini_pipeline_verified': True,
            'complete_data_format': 'ready_for_domain_mixing_training',
            'rot_mats_saved_as_rot_mats': True,
            '6d_rotations_saved_as_rot_6d': True,
            'keypoints_include_hands_feet': True,
            'visualizations_saved': SAVE_VISUALIZATIONS
        }
    }
    
    processed_frames = 0
    matched_frames = 0
    complete_data_frames = 0
    visualized_frames = 0
    
    for frame_idx in tqdm(range(num_frames), desc=f"  Processing frames"):
        valid_actors = [seq_data['campose_valid'][actor_idx][frame_idx] for actor_idx in range(num_actors)]
        if not any(valid_actors):
            continue
        
        img_path = os.path.join(IMAGE_FILES_DIR, seq_data['sequence'], f'image_{frame_idx:05d}.jpg')
        if not os.path.exists(img_path):
            continue
        
        img_original = cv2.imread(img_path)
        if img_original is None:
            continue
        
        resolution = [img_original.shape[1], img_original.shape[0]]
        
        try:
            result = next(pose_detector(img_original, show=False))
            detections_original = extract_detections_from_result(result)
        except Exception as e:
            print(f"    Warning: Detection failed for frame {frame_idx}: {e}")
            continue
        
        if SAVE_VISUALIZATIONS and detections_original and (frame_idx % VIS_EVERY_N_FRAMES == 0):
            vis_path = visualize_coco_wholebody_detections(img_original, detections_original, seq_name, frame_idx)
            visualized_frames += 1
        
        K, R, t = extract_camera_parameters(seq_data, frame_idx)
        
        gt_joints_2d_list = []
        joints_3d_centered_list = []
        root_translation_list = []
        rot_mats_list = []
        rot_6d_list = []
        cam_pose = seq_data['cam_poses'][frame_idx]
        
        for actor_idx in range(num_actors):
            if not seq_data['campose_valid'][actor_idx][frame_idx]:
                gt_joints_2d_list.append(None)
                joints_3d_centered_list.append(None)
                root_translation_list.append(None)
                rot_mats_list.append(None)
                rot_6d_list.append(None)
                continue
            
            joints_3d_world = seq_data['jointPositions'][actor_idx][frame_idx].reshape(24, 3)
            
            root_position = joints_3d_world[0].copy()
            root_translation_list.append(root_position)
            
            joints_3d_centered = joints_3d_world - joints_3d_world[0]
            joints_3d_centered_list.append(joints_3d_centered)
            
            joints_2d_smpl = project_3d_to_2d(joints_3d_world, K, cam_pose)
            
            joints_2d_coco_body = smpl_to_coco_body_subset(joints_2d_smpl)
            gt_joints_2d_list.append(joints_2d_coco_body)
            
            pose_params = seq_data['poses'][actor_idx][frame_idx]
            rot_mats, rot_6d = process_rotations_houdini_pipeline(pose_params, rest_pose_data)
            rot_mats_list.append(rot_mats)
            rot_6d_list.append(rot_6d)
        
        valid_gt = [gt for gt in gt_joints_2d_list if gt is not None]
        if valid_gt and detections_original:
            detection_keypoints = [det[0] for det in detections_original]
            matches = match_detections_to_actors(detection_keypoints, valid_gt)
        else:
            matches = [None] * len(gt_joints_2d_list)
        
        frame_detections = {}
        valid_actor_idx = 0
        frame_has_matches = False
        frame_has_complete_data = False
        
        for actor_idx in range(num_actors):
            if gt_joints_2d_list[actor_idx] is None:
                frame_detections[actor_idx] = {
                    'keypoints': None,
                    'scores': None,
                    'joints_3d_centered': None,
                    'rot_mats': None,
                    'rot_6d': None,
                    'K': None,
                    'R': None,
                    't': None,
                    'resolution': None,
                    'root_translation': None,
                    'matched': False,
                    'reason': 'invalid_campose'
                }
            else:
                joints_3d_centered = joints_3d_centered_list[actor_idx]
                root_translation = root_translation_list[actor_idx]
                rot_mats = rot_mats_list[actor_idx]
                rot_6d = rot_6d_list[actor_idx]
                
                match_idx = matches[valid_actor_idx]
                if match_idx is not None:
                    kpts_orig, scores = detections_original[match_idx]
                    frame_detections[actor_idx] = {
                        'keypoints': kpts_orig.tolist(),
                        'scores': scores.tolist(),
                        'joints_3d_centered': joints_3d_centered.tolist(),
                        'rot_mats': rot_mats.tolist(),
                        'rot_6d': rot_6d.tolist(),
                        'K': K.tolist(),
                        'R': R.tolist(),
                        't': t.tolist(),
                        'resolution': resolution,
                        'root_translation': root_translation.tolist(),
                        'matched': True,
                        'detection_idx': match_idx
                    }
                    frame_has_matches = True
                    frame_has_complete_data = True
                else:
                    frame_detections[actor_idx] = {
                        'keypoints': None,
                        'scores': None,
                        'joints_3d_centered': joints_3d_centered.tolist(),
                        'rot_mats': rot_mats.tolist(),
                        'rot_6d': rot_6d.tolist(),
                        'K': K.tolist(),
                        'R': R.tolist(),
                        't': t.tolist(),
                        'resolution': resolution,
                        'root_translation': root_translation.tolist(),
                        'matched': False,
                        'reason': 'no_match_found'
                    }
                    frame_has_complete_data = True
                valid_actor_idx += 1
        
        sequence_detections['detections'][frame_idx] = frame_detections
        processed_frames += 1
        if frame_has_matches:
            matched_frames += 1
        if frame_has_complete_data:
            complete_data_frames += 1
    
    output_file = os.path.join(DETECTIONS_DIR, f"{split}_{seq_name}_detections.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(sequence_detections, f)
    
    total_frames = len(sequence_detections['detections'])
    matched_counts = [0] * num_actors
    complete_data_counts = [0] * num_actors
    
    for frame_data in sequence_detections['detections'].values():
        for actor_idx in range(num_actors):
            if frame_data[actor_idx]['matched']:
                matched_counts[actor_idx] += 1
            if frame_data[actor_idx]['joints_3d_centered'] is not None:
                complete_data_counts[actor_idx] += 1
    
    print(f"  Saved to: {output_file}")
    print(f"  Frame statistics:")
    print(f"     Total processed frames: {total_frames}")
    print(f"     Frames with detection matches: {matched_frames} ({matched_frames/total_frames*100:.1f}%)")
    print(f"     Frames with complete GT data: {complete_data_frames} ({complete_data_frames/total_frames*100:.1f}%)")
    if SAVE_VISUALIZATIONS:
        print(f"     Visualizations saved: {visualized_frames} frames")
    
    print(f"  Actor statistics:")
    for actor_idx in range(num_actors):
        match_rate = matched_counts[actor_idx] / total_frames if total_frames > 0 else 0
        complete_rate = complete_data_counts[actor_idx] / total_frames if total_frames > 0 else 0
        print(f"     Actor {actor_idx}: detection_matches={matched_counts[actor_idx]}/{total_frames} ({match_rate:.1%}), complete_data={complete_data_counts[actor_idx]}/{total_frames} ({complete_rate:.1%})")

def main():
    print("3DPW Complete Data Preprocessing - ORIGINAL RESOLUTION")
    print("=" * 70)
    print("NO 512x512 TRANSFORMATION!")
    print("  Detecting on original image resolution")
    print("  Keeping keypoints in original pixel space")
    print("  Direct correspondence with camera parameters")
    if SAVE_VISUALIZATIONS:
        print(f"  Visualizations will be saved every {VIS_EVERY_N_FRAMES} frames")
    print()
    
    try:
        rest_pose_data = load_rest_pose_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please export rest pose data from Houdini first!")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DETECTIONS_DIR, exist_ok=True)
    if SAVE_VISUALIZATIONS:
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_detector = MMPoseInferencer(
        pose2d="rtmpose-l_8xb32-270e_coco-wholebody-384x288",
        device=device
    )
    print(f"Initialized COCO-WholeBody RTMPose detector on {device}")
    
    if not os.path.exists(SEQUENCE_FILES_DIR):
        print(f"Error: Sequence files directory not found: {SEQUENCE_FILES_DIR}")
        return
    
    all_sequences = []
    splits = ["validation"]
    
    for split in splits:
        split_dir = os.path.join(SEQUENCE_FILES_DIR, split)
        if os.path.exists(split_dir):
            sequences = [f[:-4] for f in os.listdir(split_dir) if f.endswith('.pkl')]
            for seq in sequences:
                all_sequences.append((seq, split))
            print(f"Found {len(sequences)} sequences in {split} split")
        else:
            print(f"Warning: Split directory not found: {split}")
    
    if not all_sequences:
        print("Error: No sequences found in any split!")
        return
    
    print(f"Total sequences to process: {len(all_sequences)}")
    print()
    
    processed_count = 0
    failed_count = 0
    
    for seq_name, split in all_sequences:
        try:
            process_sequence(seq_name, split, pose_detector, rest_pose_data)
            processed_count += 1
        except Exception as e:
            print(f"Error processing {seq_name} ({split}): {e}")
            failed_count += 1
            continue
    
    print(f"\n" + "="*70)
    print(f"PREPROCESSING COMPLETE!")
    print(f"="*70)
    print(f"Successfully processed: {processed_count} sequences")
    if failed_count > 0:
        print(f"Failed to process: {failed_count} sequences")
    print(f"Results saved to: {DETECTIONS_DIR}")
    if SAVE_VISUALIZATIONS:
        print(f"Visualizations saved to: {VISUALIZATIONS_DIR}")

if __name__ == "__main__":
    main()