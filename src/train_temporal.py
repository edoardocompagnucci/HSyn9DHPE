"""
Careful temporal training with extensive monitoring
"""

import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np

from data.temporal_dataset import create_temporal_dataset
from models.temporal_model import TemporalPoseModel
from utils.temporal_losses import TemporalLoss, temporal_pa_mpjpe, compute_jitter_metric
from utils.losses import pa_mpjpe_with_visibility, pck_3d_with_visibility, rotation_error_metric
from utils.transforms import NormalizerJoints2d


def main():
    # === Configuration ===

    # Resume training from checkpoint (set to experiment directory path)
    RESUME_FROM = None  # Start fresh training with 50% data and increased rotation weight

    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
    CHECKPOINT_ROOT = "checkpoints"

    # Path to pretrained single-frame model
    PRETRAINED_MODEL = None  # Set to path of best_model.pth from train.py

    # Training hyperparameters
    BATCH_SIZE = 16  # Reduced for 8GB VRAM (was 32)
    ACCUMULATION_STEPS = 4  # Increased to maintain effective batch size of 64
    DROPOUT_RATE = 0.1
    LEARNING_RATE = 1e-4  # Lower LR for temporal fine-tuning
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    NUM_JOINTS = 24

    # Temporal parameters
    SEQUENCE_LENGTH = 16  # 16 frames at 30fps = 0.5 seconds
    STRIDE = 16  # No overlap (standard practice, reduces redundancy)
    NUM_TEMPORAL_LAYERS = 2  # Keep it simple
    TEMPORAL_KERNEL_SIZE = 3
    USE_DILATION = True

    # Loss weights
    LOSS_POS_WEIGHT = 1.0
    LOSS_ROT_WEIGHT = 0.007  # Slightly increased from 0.005 (single-frame used 0.005)
    LOSS_VELOCITY_WEIGHT = 0.08  # Slightly increased from 0.05 for smoother motion
    LOSS_ACCELERATION_WEIGHT = 0.03  # Slightly increased from 0.02 for better jitter reduction

    # Training control
    WARMUP_EPOCHS = 5
    VALIDATE_EVERY_N_EPOCHS = 1
    CHECKPOINT_EVERY_N_EPOCHS = 1
    EARLY_STOPPING_PATIENCE = 20

    # Augmentation (match train.py exactly)
    NOISE_STD_BASE = 0.08  # Match train.py
    NOISE_PROB = 0.7  # Match train.py

    # === Setup ===

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Data normalization
    normalizer = NormalizerJoints2d()

    # Create datasets
    print("Creating datasets...")
    train_dataset = create_temporal_dataset(
        data_root=DATA_ROOT,
        dataset_type='synthetic',
        split='train',
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
        transform=normalizer,
        use_augmentation=True,
        noise_std=NOISE_STD_BASE,
        noise_prob=NOISE_PROB
    )

    # Use 3DPW for validation (real detector data)
    val_dataset = create_temporal_dataset(
        data_root=DATA_ROOT,
        dataset_type='real',
        split='val',
        sequence_length=SEQUENCE_LENGTH,
        stride=SEQUENCE_LENGTH,  # No overlap for validation
        transform=normalizer,
        confidence_threshold=0.3
    )

    # Create data loaders
    # Note: num_workers=0 on Windows due to NPZ cache pickling issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} sequences ({len(train_loader)} batches)")
    print(f"  Val: {len(val_dataset)} sequences ({len(val_loader)} batches)")

    # Create model
    print(f"\nCreating temporal model...")
    model = TemporalPoseModel(
        num_joints=NUM_JOINTS,
        dim=384,
        depth=8,
        heads=12,
        ffn_dim=1536,
        dropout=DROPOUT_RATE,
        num_temporal_layers=NUM_TEMPORAL_LAYERS,
        temporal_kernel_size=TEMPORAL_KERNEL_SIZE,
        use_dilation=USE_DILATION
    ).to(device)

    # Load pretrained spatial encoder if available
    if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL):
        model.load_spatial_encoder(PRETRAINED_MODEL)
        print(f"  Loaded pretrained spatial encoder: {PRETRAINED_MODEL}")
    else:
        print(f"  No pretrained model found, training from scratch")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create loss function
    criterion = TemporalLoss(
        position_weight=LOSS_POS_WEIGHT,
        rotation_weight=LOSS_ROT_WEIGHT,
        velocity_weight=LOSS_VELOCITY_WEIGHT,
        acceleration_weight=LOSS_ACCELERATION_WEIGHT,
        use_geodesic=True
    )

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

    # Create or resume experiment directory
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

    start_epoch = 1
    best_val_pa_mpjpe = float('inf')
    best_val_jitter = float('inf')
    best_epoch = 0
    early_stopping_counter = 0

    # Handle checkpoint resuming
    if RESUME_FROM and os.path.isdir(RESUME_FROM):
        experiment_dir = RESUME_FROM
        print(f"\nResuming from experiment: {experiment_dir}")

        # Find latest checkpoint
        checkpoint_files = []
        for f in os.listdir(experiment_dir):
            if f.startswith('checkpoint_epoch_') and f.endswith('.pth'):
                epoch_num = int(f.split('_')[2].split('.')[0])
                checkpoint_files.append((epoch_num, os.path.join(experiment_dir, f)))

        if checkpoint_files:
            checkpoint_files.sort(reverse=True)
            last_epoch, checkpoint_path = checkpoint_files[0]

            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            # Load BEST metrics (not current checkpoint metrics)
            best_val_pa_mpjpe = checkpoint.get('best_val_pa_mpjpe', checkpoint.get('val_pa_mpjpe', float('inf')))
            best_val_jitter = checkpoint.get('best_val_jitter', checkpoint.get('val_jitter', float('inf')))
            best_epoch = checkpoint.get('best_epoch', checkpoint['epoch'])
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0)

            print(f"Resumed from epoch {checkpoint['epoch']}")
            print(f"Best PA-MPJPE: {best_val_pa_mpjpe*1000:.1f}mm at epoch {best_epoch}")
            print(f"Best jitter: {best_val_jitter:.1f}mm/s²")
            print(f"Early stopping counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")
        else:
            print(f"No checkpoints found in {experiment_dir}, starting fresh")
    else:
        exp_name = f"temporal_{datetime.now():%Y%m%d_%H%M%S}"
        experiment_dir = os.path.join(CHECKPOINT_ROOT, exp_name)
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"\nExperiment directory: {experiment_dir}")

    # === Training Loop ===

    print(f"\nStarting training...\n")
    print("="*80)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Learning rate warmup
        if epoch <= WARMUP_EPOCHS:
            lr = LEARNING_RATE * (epoch / WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # === Training Phase ===
        model.train()

        train_losses = {
            'total': 0.0,
            'position': 0.0,
            'rotation': 0.0,
            'velocity': 0.0,
            'acceleration': 0.0,
            'mpjpe': 0.0,
            'jitter': 0.0
        }

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            joints_2d = batch['joints_2d'].to(device)  # [B, T, J, 2]
            joints_3d = batch['joints_3d_centered'].to(device)  # [B, T, J, 3]
            rot_6d = batch['rot_6d'].to(device)  # [B, T, J, 6]
            confidence = batch['confidence'].to(device)  # [B, T, J]
            visibility = batch['visibility'].to(device)  # [B, T, J]

            B, T, J, _ = joints_2d.shape

            # Forward pass
            pred_pos, pred_rot = model(joints_2d)  # [B, T, J*3], [B, T, J, 6]

            # Reshape targets
            target_pos = joints_3d.reshape(B, T, -1)  # [B, T, J*3]

            # Compute loss
            loss, loss_dict = criterion(
                pred_pos, target_pos,
                pred_rot, rot_6d,
                confidence=confidence,
                visibility=visibility
            )

            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Accumulate losses
            for key in train_losses.keys():
                if key in loss_dict:
                    train_losses[key] += loss_dict[key].item() if torch.is_tensor(loss_dict[key]) else loss_dict[key]

        # Average training losses
        for key in train_losses.keys():
            train_losses[key] /= len(train_loader)

        # === Validation Phase ===
        if epoch % VALIDATE_EVERY_N_EPOCHS == 0:
            model.eval()

            val_losses = {
                'total': 0.0,
                'position': 0.0,
                'mpjpe': 0.0,
                'pa_mpjpe': 0.0,
                'jitter': 0.0,
                'rotation_error': 0.0
            }

            with torch.no_grad():
                for batch in val_loader:
                    joints_2d = batch['joints_2d'].to(device)
                    joints_3d = batch['joints_3d_centered'].to(device)
                    rot_6d = batch['rot_6d'].to(device)
                    confidence = batch['confidence'].to(device)
                    visibility = batch['visibility'].to(device)

                    B, T, J, _ = joints_2d.shape

                    # Forward
                    pred_pos, pred_rot = model(joints_2d)
                    target_pos = joints_3d.reshape(B, T, -1)

                    # Compute metrics
                    loss, loss_dict = criterion(
                        pred_pos, target_pos,
                        pred_rot, rot_6d,
                        confidence=confidence,
                        visibility=visibility
                    )

                    # PA-MPJPE (per-frame alignment)
                    pa_mpjpe = temporal_pa_mpjpe(pred_pos, target_pos, visibility, confidence)

                    # Jitter
                    jitter = compute_jitter_metric(pred_pos, fps=30.0)

                    # Rotation error (average across frames)
                    rot_errors = []
                    for t in range(T):
                        rot_err, _ = rotation_error_metric(pred_rot[:, t], rot_6d[:, t], visibility[:, t])
                        rot_errors.append(rot_err)
                    avg_rot_error = torch.stack(rot_errors).mean()

                    val_losses['total'] += loss_dict['total'].item()
                    val_losses['position'] += loss_dict['position'].item()
                    val_losses['mpjpe'] += loss_dict['mpjpe'].item()
                    val_losses['pa_mpjpe'] += pa_mpjpe.item()
                    val_losses['jitter'] += jitter.item()
                    val_losses['rotation_error'] += avg_rot_error.item()

            # Average validation losses
            for key in val_losses.keys():
                val_losses[key] /= len(val_loader)

            # Learning rate scheduling
            if epoch > WARMUP_EPOCHS:
                scheduler.step(val_losses['pa_mpjpe'])

            # Print progress
            print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
            print(f"  Train - MPJPE: {train_losses['mpjpe']*1000:.1f}mm, Jitter: {train_losses['jitter']:.1f}mm/s²")
            print(f"          Position: {train_losses['position']:.4f}, Vel: {train_losses['velocity']:.4f}, Acc: {train_losses['acceleration']:.4f}")
            print(f"  Val   - MPJPE: {val_losses['mpjpe']*1000:.1f}mm, PA-MPJPE: {val_losses['pa_mpjpe']*1000:.1f}mm")
            print(f"          Jitter: {val_losses['jitter']:.1f}mm/s², Rot Error: {val_losses['rotation_error']:.2f}°")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Save best model
            is_best = val_losses['pa_mpjpe'] < best_val_pa_mpjpe

            if is_best:
                best_val_pa_mpjpe = val_losses['pa_mpjpe']
                best_val_jitter = val_losses['jitter']
                best_epoch = epoch
                early_stopping_counter = 0

                best_model_path = os.path.join(experiment_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_mpjpe': val_losses['mpjpe'],
                    'val_pa_mpjpe': val_losses['pa_mpjpe'],
                    'val_jitter': val_losses['jitter'],
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'config': {
                        'sequence_length': SEQUENCE_LENGTH,
                        'num_temporal_layers': NUM_TEMPORAL_LAYERS,
                        'temporal_kernel_size': TEMPORAL_KERNEL_SIZE,
                        'dropout': DROPOUT_RATE,
                        'loss_weights': {
                            'position': LOSS_POS_WEIGHT,
                            'rotation': LOSS_ROT_WEIGHT,
                            'velocity': LOSS_VELOCITY_WEIGHT,
                            'acceleration': LOSS_ACCELERATION_WEIGHT
                        }
                    }
                }, best_model_path)

                print(f"  [BEST] New best PA-MPJPE: {val_losses['pa_mpjpe']*1000:.1f}mm, Jitter: {val_losses['jitter']:.1f}mm/s²")
            else:
                early_stopping_counter += 1
                print(f"  Best: PA-MPJPE={best_val_pa_mpjpe*1000:.1f}mm (epoch {best_epoch}), ES: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")

            # Early stopping
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

            # Save checkpoint
            if epoch % CHECKPOINT_EVERY_N_EPOCHS == 0:
                ckpt_path = os.path.join(experiment_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_pa_mpjpe': val_losses['pa_mpjpe'],  # Current epoch metrics
                    'val_jitter': val_losses['jitter'],
                    'best_val_pa_mpjpe': best_val_pa_mpjpe,  # Best metrics so far
                    'best_val_jitter': best_val_jitter,
                    'best_epoch': best_epoch,
                    'early_stopping_counter': early_stopping_counter
                }, ckpt_path)

        print("-"*80)

    # === Training Complete ===
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best validation metrics (epoch {best_epoch}):")
    print(f"  PA-MPJPE: {best_val_pa_mpjpe*1000:.1f}mm")
    print(f"  Jitter: {best_val_jitter:.1f}mm/s²")
    print(f"\nBest model saved to: {os.path.join(experiment_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
