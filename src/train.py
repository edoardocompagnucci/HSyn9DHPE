import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np
import glob

from data.mixed_pose_dataset import create_dataset
from models.graphformer import GraphFormerPose
from utils.losses import (
    combined_pose_bone_projection_loss_with_visibility,
    pa_mpjpe_with_visibility,
    pck_3d_with_visibility,
    rotation_error_metric
)
from utils.transforms import NormalizerJoints2d


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the experiment directory"""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None, 0
    
    # Extract epoch numbers and find the latest
    epochs = []
    for ckpt in checkpoint_files:
        try:
            epoch_num = int(ckpt.split("epoch_")[1].split(".pth")[0])
            epochs.append((epoch_num, ckpt))
        except:
            continue
    
    if not epochs:
        return None, 0
    
    # Return path to latest checkpoint and epoch number
    latest = max(epochs, key=lambda x: x[0])
    return latest[1], latest[0]


def warmup_lr(epoch, base_lr=1e-5, warmup_epochs=5):
    """Learning rate warmup for first few epochs"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


def compute_joint_group_errors(pred_pos, target_pos, visibility_mask=None):
    """Compute errors grouped by anatomical regions"""
    batch_size = pred_pos.shape[0]
    pred_3d = pred_pos.reshape(batch_size, 24, 3)
    target_3d = target_pos.reshape(batch_size, 24, 3)
    
    joint_groups = {
        'pelvis': [0],
        'torso': [3, 6, 9, 12],
        'head': [15],
        'arms': [13, 14, 16, 17, 18, 19],
        'hands': [20, 21, 22, 23],
        'legs': [1, 2, 4, 5, 7, 8],
        'feet': [10, 11]
    }
    
    group_errors = {}
    for group_name, joint_indices in joint_groups.items():
        joint_errors = []
        for idx in joint_indices:
            if idx < pred_3d.shape[1]:
                error = torch.sqrt(torch.sum((pred_3d[:, idx] - target_3d[:, idx]) ** 2, dim=1))
                
                if visibility_mask is not None:
                    visible_joints = visibility_mask[:, idx]
                    if visible_joints.any():
                        error = error[visible_joints]
                    else:
                        continue
                
                joint_errors.append(error.mean().item() * 1000)
        
        group_errors[group_name] = np.mean(joint_errors) if joint_errors else 0.0
    
    return group_errors


def main():
    RESUME_FROM = None
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    CHECKPOINT_ROOT = "checkpoints"

    # Training hyperparameters
    BATCH_SIZE = 64
    ACCUMULATION_STEPS = 2
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 8e-5
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 400
    NUM_JOINTS = 24

    # Learning rate warmup
    USE_WARMUP = True
    WARMUP_EPOCHS = 10

    # Loss weights
    LOSS_POS_WEIGHT = 1.0
    LOSS_ROT_WEIGHT = 0.005
    LOSS_BONE_WEIGHT = 0.0
    LOSS_PROJECTION_WEIGHT = 0.0

    # Augmentation
    USE_2D_NOISE_AUG = True
    NOISE_STD_BASE = 0.08
    NOISE_PROB = 0.7

    # Confidence handling
    CONFIDENCE_THRESHOLD = 0.3
    CONFIDENCE_MODE = 'soft'

    # Training control
    early_stopping_patience = 40
    early_stopping_counter = 0
    CHECKPOINT_EVERY_N_EPOCHS = 1
    EVAL_EVERY_N_EPOCHS = 5
    VALIDATE_EVERY_N_EPOCHS = 1

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data normalization
    normalizer = NormalizerJoints2d()
    
    # Create datasets
    train_dataset = create_dataset(
        data_root=DATA_ROOT,
        dataset_type='synthetic',
        split='train',
        transform=normalizer,
        skip_invisible=False,
        use_2d_noise_aug=USE_2D_NOISE_AUG,
        noise_std_base=NOISE_STD_BASE,
        noise_prob=NOISE_PROB
    )
    
    val_dataset = create_dataset(
        data_root=DATA_ROOT,
        dataset_type='real',
        split='val',
        transform=normalizer,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        confidence_mode=CONFIDENCE_MODE
    )
    
    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,   
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2,  # Keep low for validation
        pin_memory=True,
        persistent_workers=False  # Safer for NPZ files
    )
    
    # Create model
    model = GraphFormerPose(
        num_joints=NUM_JOINTS,
        dim=384,
        depth=8,
        heads=12,
        ffn_dim=1536,
        dropout=DROPOUT_RATE
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: GraphFormer with {total_params:,} parameters")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5, verbose=True
    )

    # Initialize training state
    start_epoch = 1
    best_val_mpjpe = float("inf")
    best_val_pa_mpjpe = float("inf")
    best_epoch = 0
    training_history = []
    
    # Handle checkpoint resuming
    if RESUME_FROM and os.path.isdir(RESUME_FROM):
        experiment_dir = RESUME_FROM
        checkpoint_path, last_epoch = find_latest_checkpoint(experiment_dir)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            start_epoch = checkpoint['epoch'] + 1
            best_val_mpjpe = checkpoint.get('best_val_mpjpe', float('inf'))
            best_val_pa_mpjpe = checkpoint.get('best_val_pa_mpjpe', float('inf'))
            best_epoch = checkpoint.get('best_epoch', 0)
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
            training_history = checkpoint.get('training_history', [])
            
            print(f"Resumed from epoch {checkpoint['epoch']}")
            print(f"Best PA-MPJPE so far: {best_val_pa_mpjpe:.2f}mm at epoch {best_epoch}")
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
        else:
            print(f"No valid checkpoint found in {experiment_dir}, starting fresh")
            os.makedirs(experiment_dir, exist_ok=True)
    else:
        # Create new experiment directory
        os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
        exp_name = f"train_{datetime.now():%Y%m%d_%H%M%S}_graphformer"
        experiment_dir = os.path.join(CHECKPOINT_ROOT, exp_name)
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"New experiment directory: {experiment_dir}")
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Apply learning rate warmup
        if USE_WARMUP and epoch <= WARMUP_EPOCHS:
            current_lr = warmup_lr(epoch - 1, LEARNING_RATE, WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # Training phase
        model.train()
        running_train_mpjpe = 0.0
        running_pos_loss = 0.0
        running_rot_loss = 0.0
        running_bone_loss = 0.0
        running_proj_loss = 0.0
        running_pixel_error = 0.0
        
        train_group_errors_sum = {group: 0.0 for group in ['pelvis', 'torso', 'head', 'arms', 'hands', 'legs', 'feet']}
        train_batches_for_groups = 0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["joints_2d"].to(device)
            target_pos = batch["joints_3d_centered"].to(device)
            target_rot = batch["rot_6d"].to(device)
            
            # Get visibility information
            visibility_mask = batch.get("visibility_mask")
            confidence_weights = batch.get("confidence_weights")
            
            if visibility_mask is not None:
                visibility_mask = visibility_mask.to(device)
            if confidence_weights is not None:
                confidence_weights = confidence_weights.to(device)
            
            # Extract camera parameters and root translation
            camera_params = None
            root_translation = None
            
            if all(key in batch for key in ['K', 'R', 't', 'resolution', 'root_translation']):
                camera_params = {
                    'K': batch['K'].to(device),
                    'R': batch['R'].to(device),
                    't': batch['t'].to(device),
                    'resolution': batch['resolution'].to(device)
                }
                root_translation = batch['root_translation'].to(device)

            # Forward pass
            pos3d, rot6d = model(inputs)
            
            # Flatten pos3d to match expected format
            if pos3d.dim() == 3:
                pos3d = pos3d.view(pos3d.shape[0], -1)

            # Prepare dictionaries for loss computation
            pred_dict = {
                'positions': pos3d,
                'rotations': rot6d
            }
            target_dict = {
                'positions': target_pos.flatten(1),
                'rotations': target_rot,
                'joints_2d': inputs
            }

            # Compute loss
            loss_dict = combined_pose_bone_projection_loss_with_visibility(
                pred_dict, target_dict,
                camera_params=camera_params,
                root_translation=root_translation,
                visibility_mask=visibility_mask,
                confidence_weights=confidence_weights,
                pos_weight=LOSS_POS_WEIGHT, 
                rot_weight=LOSS_ROT_WEIGHT,
                bone_weight=LOSS_BONE_WEIGHT,
                projection_weight=LOSS_PROJECTION_WEIGHT,
                use_geodesic=True,
                use_extremity_weights=True
            )
            loss = loss_dict['total'] / ACCUMULATION_STEPS

            # Backward pass with gradient accumulation
            loss.backward()
            
            # Step optimizer after accumulation
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Accumulate losses
            running_train_mpjpe += loss_dict['mpjpe'].item()
            running_pos_loss += loss_dict['position'].item()
            running_rot_loss += loss_dict['rotation'].item()
            running_bone_loss += loss_dict['bone'].item()
            running_proj_loss += loss_dict['projection'].item()
            running_pixel_error += loss_dict['pixel_error'].item()
            
            # Compute joint group errors periodically
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    group_errors = compute_joint_group_errors(pos3d, target_pos.flatten(1), visibility_mask)
                    for group_name, error in group_errors.items():
                        train_group_errors_sum[group_name] += error
                    train_batches_for_groups += 1
        
        # Handle remaining gradients
        if len(train_loader) % ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Average training losses
        avg_train_mpjpe = running_train_mpjpe / len(train_loader)
        avg_train_pos = running_pos_loss / len(train_loader)
        avg_train_rot = running_rot_loss / len(train_loader)
        avg_train_bone = running_bone_loss / len(train_loader)
        avg_train_proj = running_proj_loss / len(train_loader)
        avg_train_pixel_error = running_pixel_error / len(train_loader)
        
        # Average joint group errors
        if train_batches_for_groups > 0:
            for group_name in train_group_errors_sum:
                train_group_errors_sum[group_name] /= train_batches_for_groups

        # Validation phase
        if epoch % VALIDATE_EVERY_N_EPOCHS == 0:
            model.eval()
            running_val_mpjpe = 0.0
            running_val_pa_mpjpe = 0.0
            running_val_pos = 0.0
            running_val_rot = 0.0
            running_val_bone = 0.0
            running_val_proj = 0.0
            running_val_pixel_error = 0.0
            running_val_rot_error = 0.0
            
            # PCK accumulators
            pck_50_sum = 0.0
            pck_100_sum = 0.0
            pck_150_sum = 0.0
            pck_batches = 0
            
            # Validation joint group errors
            val_group_errors_sum = {group: 0.0 for group in ['pelvis', 'torso', 'head', 'arms', 'hands', 'legs', 'feet']}
            val_batches_for_groups = 0
            
            # Per-joint rotation errors
            per_joint_rot_errors_sum = torch.zeros(NUM_JOINTS, device=device)
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    inputs = batch["joints_2d"].to(device)
                    target_pos = batch["joints_3d_centered"].to(device)
                    target_rot = batch["rot_6d"].to(device)
                    
                    # Get visibility information
                    visibility_mask = batch.get("visibility_mask")
                    confidence_weights = batch.get("confidence_weights")
                    
                    if visibility_mask is not None:
                        visibility_mask = visibility_mask.to(device)
                    if confidence_weights is not None:
                        confidence_weights = confidence_weights.to(device)
                    
                    # Extract camera params
                    camera_params = None
                    root_translation = None
                    
                    if all(key in batch for key in ['K', 'R', 't', 'resolution', 'root_translation']):
                        camera_params = {
                            'K': batch['K'].to(device),
                            'R': batch['R'].to(device),
                            't': batch['t'].to(device),
                            'resolution': batch['resolution'].to(device)
                        }
                        root_translation = batch['root_translation'].to(device)
                    
                    pos3d, rot6d = model(inputs)
                    
                    if pos3d.dim() == 3:
                        pos3d = pos3d.view(pos3d.shape[0], -1)
                    
                    pred_dict = {
                        'positions': pos3d,
                        'rotations': rot6d
                    }
                    target_dict = {
                        'positions': target_pos.flatten(1),
                        'rotations': target_rot,
                        'joints_2d': inputs
                    }
                    
                    loss_dict = combined_pose_bone_projection_loss_with_visibility(
                        pred_dict, target_dict,
                        camera_params=camera_params,
                        root_translation=root_translation,
                        visibility_mask=visibility_mask,
                        confidence_weights=confidence_weights,
                        pos_weight=LOSS_POS_WEIGHT, 
                        rot_weight=LOSS_ROT_WEIGHT,
                        bone_weight=LOSS_BONE_WEIGHT,
                        projection_weight=LOSS_PROJECTION_WEIGHT,
                        use_geodesic=True,
                        use_extremity_weights=True
                    )
                    
                    # Compute PA-MPJPE
                    pa_mpjpe = pa_mpjpe_with_visibility(
                        pos3d, target_pos.flatten(1), 
                        visibility_mask, confidence_weights
                    )
                    
                    # Compute PCK
                    pck_results = pck_3d_with_visibility(
                        pos3d, target_pos.flatten(1),
                        thresholds=[50, 100, 150],
                        visibility_mask=visibility_mask,
                        confidence_weights=confidence_weights
                    )
                    
                    # Compute rotation error
                    rot_error, per_joint_rot = rotation_error_metric(
                        rot6d, target_rot, visibility_mask
                    )
                    
                    running_val_mpjpe += loss_dict['mpjpe'].item()
                    running_val_pa_mpjpe += pa_mpjpe.item()
                    running_val_pos += loss_dict['position'].item()
                    running_val_rot += loss_dict['rotation'].item()
                    running_val_bone += loss_dict['bone'].item()
                    running_val_proj += loss_dict['projection'].item()
                    running_val_pixel_error += loss_dict['pixel_error'].item()
                    running_val_rot_error += rot_error.item()
                    
                    # Accumulate PCK
                    pck_50_sum += pck_results['pck_50']
                    pck_100_sum += pck_results['pck_100']
                    pck_150_sum += pck_results['pck_150']
                    pck_batches += 1
                    
                    # Accumulate per-joint rotation errors
                    per_joint_rot_errors_sum += per_joint_rot
                    
                    # Compute joint group errors
                    if batch_idx % 10 == 0:
                        group_errors = compute_joint_group_errors(pos3d, target_pos.flatten(1), visibility_mask)
                        for group_name, error in group_errors.items():
                            val_group_errors_sum[group_name] += error
                        val_batches_for_groups += 1

            # Average validation losses
            avg_val_mpjpe = running_val_mpjpe / len(val_loader)
            avg_val_pa_mpjpe = running_val_pa_mpjpe / len(val_loader)
            avg_val_pos = running_val_pos / len(val_loader)
            avg_val_rot = running_val_rot / len(val_loader)
            avg_val_bone = running_val_bone / len(val_loader)
            avg_val_proj = running_val_proj / len(val_loader)
            avg_val_pixel_error = running_val_pixel_error / len(val_loader)
            avg_val_rot_error = running_val_rot_error / len(val_loader)
            
            # Average PCK
            avg_pck_50 = pck_50_sum / pck_batches if pck_batches > 0 else 0
            avg_pck_100 = pck_100_sum / pck_batches if pck_batches > 0 else 0
            avg_pck_150 = pck_150_sum / pck_batches if pck_batches > 0 else 0
            
            # Average per-joint rotation errors
            avg_per_joint_rot_errors = per_joint_rot_errors_sum / len(val_loader)
            
            # Average joint group errors
            if val_batches_for_groups > 0:
                for group_name in val_group_errors_sum:
                    val_group_errors_sum[group_name] /= val_batches_for_groups
        else:
            # If not validating, use previous values
            avg_val_mpjpe = best_val_mpjpe if best_val_mpjpe != float("inf") else 1.0
            avg_val_pa_mpjpe = best_val_pa_mpjpe if best_val_pa_mpjpe != float("inf") else 1.0
            avg_val_pos = avg_val_rot = avg_val_bone = avg_val_proj = 0.0
            avg_val_pixel_error = 0.0
            avg_val_rot_error = 0.0
            avg_pck_50 = avg_pck_100 = avg_pck_150 = 0.0
            val_group_errors_sum = train_group_errors_sum

        # Learning rate scheduling
        if not USE_WARMUP or epoch > WARMUP_EPOCHS:
            scheduler.step(avg_val_pa_mpjpe)
        current_lr = optimizer.param_groups[0]['lr']

        # Convert to mm for reporting
        train_mpjpe_mm = avg_train_mpjpe * 1000.0
        val_mpjpe_mm = avg_val_mpjpe * 1000.0 if epoch % VALIDATE_EVERY_N_EPOCHS == 0 else best_val_mpjpe * 1000.0
        val_pa_mpjpe_mm = avg_val_pa_mpjpe * 1000.0 if epoch % VALIDATE_EVERY_N_EPOCHS == 0 else best_val_pa_mpjpe * 1000.0

        # Print progress
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"  Train - MPJPE: {train_mpjpe_mm:.1f}mm, Pixel Error: {avg_train_pixel_error:.1f}px")
        
        if epoch % VALIDATE_EVERY_N_EPOCHS == 0:
            print(f"  Val   - MPJPE: {val_mpjpe_mm:.1f}mm, PA-MPJPE: {val_pa_mpjpe_mm:.1f}mm, Rot Error: {avg_val_rot_error:.2f}deg")
            print(f"          PCK@50mm: {avg_pck_50:.1f}%, PCK@100mm: {avg_pck_100:.1f}%, PCK@150mm: {avg_pck_150:.1f}%")
            print(f"          Pixel Error: {avg_val_pixel_error:.1f}px")
            print(f"          Domain gap: {val_mpjpe_mm - train_mpjpe_mm:.1f}mm")

        # Detailed evaluation every N epochs
        if epoch % EVAL_EVERY_N_EPOCHS == 0 and epoch % VALIDATE_EVERY_N_EPOCHS == 0:
            print(f"\nDetailed evaluation at epoch {epoch}:")
            print(f"  Best validation so far: PA-MPJPE={best_val_pa_mpjpe*1000:.1f}mm, MPJPE={best_val_mpjpe*1000:.1f}mm (epoch {best_epoch})")
            
            # Print joint group errors
            print("\n  Performance by anatomical region:")
            print("                    Training    Validation")
            for group in ['pelvis', 'torso', 'head', 'arms', 'hands', 'legs', 'feet']:
                print(f"    {group:10s}: {train_group_errors_sum[group]:7.1f}mm  {val_group_errors_sum[group]:7.1f}mm")
            
            # Print per-joint rotation errors for key joints
            if epoch % VALIDATE_EVERY_N_EPOCHS == 0:
                print("\n  Rotation errors for key joints (degrees):")
                key_joints = {
                    'Pelvis': 0, 'L_Hip': 1, 'R_Hip': 2, 'L_Knee': 4, 'R_Knee': 5,
                    'L_Shoulder': 16, 'R_Shoulder': 17, 'L_Elbow': 18, 'R_Elbow': 19,
                    'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand': 22, 'R_Hand': 23
                }
                for joint_name, joint_idx in key_joints.items():
                    print(f"    {joint_name:12s}: {avg_per_joint_rot_errors[joint_idx]:.2f}deg")

        # Save best model
        if epoch % VALIDATE_EVERY_N_EPOCHS == 0:
            is_best = False
            if avg_val_pa_mpjpe < best_val_pa_mpjpe:
                is_best = True
                best_val_pa_mpjpe = avg_val_pa_mpjpe
                best_val_mpjpe = avg_val_mpjpe
                best_epoch = epoch
            
            if is_best:
                early_stopping_counter = 0
                
                best_model_path = os.path.join(experiment_dir, "best_model.pth")
                torch.save({
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_mpjpe": best_val_mpjpe,
                    "val_pa_mpjpe": best_val_pa_mpjpe,
                    "val_rot_error": avg_val_rot_error,
                    "val_pck": {
                        "pck_50": avg_pck_50,
                        "pck_100": avg_pck_100,
                        "pck_150": avg_pck_150
                    },
                    "train_mpjpe": avg_train_mpjpe,
                    "best_epoch": best_epoch,
                    "best_val_mpjpe": best_val_mpjpe,
                    "best_val_pa_mpjpe": best_val_pa_mpjpe,
                    "early_stopping_counter": early_stopping_counter,
                    "training_history": training_history,
                    "model_type": "graphformer",
                    "model_config": {
                        "dim": 384,
                        "depth": 8,
                        "heads": 12,
                        "dropout": DROPOUT_RATE
                    },
                    "group_errors": val_group_errors_sum,
                    "train_group_errors": train_group_errors_sum,
                    "per_joint_rot_errors": avg_per_joint_rot_errors.cpu().numpy().tolist(),
                    "confidence_handling": {
                        "mode": CONFIDENCE_MODE,
                        "threshold": CONFIDENCE_THRESHOLD
                    }
                }, best_model_path)
                
                print(f"  New best validation PA-MPJPE: {val_pa_mpjpe_mm:.1f}mm")
                print(f"  Saved best model")
            else:
                early_stopping_counter += 1
                
                if early_stopping_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
            
            # Store history
            if epoch % VALIDATE_EVERY_N_EPOCHS == 0:
                training_history.append({
                    'epoch': epoch,
                    'train_mpjpe': avg_train_mpjpe,
                    'val_mpjpe': avg_val_mpjpe,
                    'val_pa_mpjpe': avg_val_pa_mpjpe,
                    'val_rot_error': avg_val_rot_error,
                    'pck_50': avg_pck_50,
                    'pck_100': avg_pck_100,
                    'pck_150': avg_pck_150
                })

        # Save checkpoint at intervals
        if epoch % CHECKPOINT_EVERY_N_EPOCHS == 0:
            checkpoint_path = os.path.join(experiment_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_mpjpe": avg_val_mpjpe if epoch % VALIDATE_EVERY_N_EPOCHS == 0 else best_val_mpjpe,
                "val_pa_mpjpe": avg_val_pa_mpjpe if epoch % VALIDATE_EVERY_N_EPOCHS == 0 else best_val_pa_mpjpe,
                "val_rot_error": avg_val_rot_error if epoch % VALIDATE_EVERY_N_EPOCHS == 0 else 0,
                "val_pck": {
                    "pck_50": avg_pck_50,
                    "pck_100": avg_pck_100,
                    "pck_150": avg_pck_150
                } if epoch % VALIDATE_EVERY_N_EPOCHS == 0 else None,
                "best_epoch": best_epoch,
                "best_val_mpjpe": best_val_mpjpe,
                "best_val_pa_mpjpe": best_val_pa_mpjpe,
                "early_stopping_counter": early_stopping_counter,
                "training_history": training_history,
                "model_type": "graphformer",
                "model_config": {
                    "dim": 384,
                    "depth": 8,
                    "heads": 12,
                    "dropout": DROPOUT_RATE
                }
            }, checkpoint_path)
            print(f"  Saved checkpoint")

    # Training complete
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"{'='*70}")
    print(f"Best validation metrics (epoch {best_epoch}):")
    print(f"  PA-MPJPE: {best_val_pa_mpjpe*1000.0:.1f}mm")
    print(f"  MPJPE: {best_val_mpjpe*1000.0:.1f}mm")
    
    # Load best model to get final metrics
    best_checkpoint = torch.load(os.path.join(experiment_dir, "best_model.pth"))
    if "val_rot_error" in best_checkpoint:
        print(f"  Rotation Error: {best_checkpoint['val_rot_error']:.2f}deg")
    if "val_pck" in best_checkpoint:
        print(f"  PCK@50mm: {best_checkpoint['val_pck']['pck_50']:.1f}%")
        print(f"  PCK@100mm: {best_checkpoint['val_pck']['pck_100']:.1f}%")
        print(f"  PCK@150mm: {best_checkpoint['val_pck']['pck_150']:.1f}%")

    # Save final model
    final_path = os.path.join(experiment_dir, "final_model.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_val_mpjpe": best_val_mpjpe,
        "best_val_pa_mpjpe": best_val_pa_mpjpe,
        "best_epoch": best_epoch,
        "model_type": "graphformer",
        "model_config": {
            "dim": 384,
            "depth": 8,
            "heads": 12,
            "dropout": DROPOUT_RATE
        },
        "confidence_handling": {
            "mode": CONFIDENCE_MODE,
            "threshold": CONFIDENCE_THRESHOLD
        }
    }, final_path)
    
    print(f"\nSaved final model -> {final_path}")


if __name__ == "__main__":
    main()