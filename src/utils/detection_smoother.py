"""
2D Detection Smoothing and Outlier Filtering
Handles bad detector outputs before 3D lifting
"""

import numpy as np
from typing import List, Tuple, Optional


class DetectionSmoother:
    """
    Smooth and filter 2D detections to handle detector failures

    Methods:
    - Outlier detection (large jumps)
    - Confidence-based filtering
    - Temporal interpolation
    - Optional Kalman filtering
    """

    def __init__(self,
                 max_displacement_px: float = 100.0,
                 min_confidence: float = 0.2,
                 use_kalman: bool = False,
                 smoothing_window: int = 5):
        """
        Args:
            max_displacement_px: Maximum allowed displacement between frames (pixels)
            min_confidence: Minimum confidence threshold for accepting detections
            use_kalman: Whether to use Kalman filtering (more complex)
            smoothing_window: Window size for moving average smoothing
        """
        self.max_displacement_px = max_displacement_px
        self.min_confidence = min_confidence
        self.use_kalman = use_kalman
        self.smoothing_window = smoothing_window

    def smooth_detections(self,
                          joints_2d_list: List[np.ndarray],
                          confidence_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[bool]]:
        """
        Main smoothing pipeline

        Args:
            joints_2d_list: List of [24, 2] arrays (one per frame)
            confidence_list: List of [24] arrays (confidence per joint)

        Returns:
            smoothed_joints: List of smoothed [24, 2] arrays
            bad_frames: List of booleans indicating which frames were corrected
        """

        if len(joints_2d_list) == 0:
            return [], []

        # Convert to numpy arrays
        joints_sequence = np.array(joints_2d_list)  # [T, 24, 2]
        confidence_sequence = np.array(confidence_list)  # [T, 24]

        T, J, _ = joints_sequence.shape

        # Step 1: Detect bad frames
        bad_frames = self._detect_bad_frames(joints_sequence, confidence_sequence)

        # Step 2: Interpolate bad frames
        smoothed_sequence = self._interpolate_bad_frames(joints_sequence, bad_frames)

        # Step 3: Apply temporal smoothing
        if self.use_kalman:
            smoothed_sequence = self._kalman_smooth(smoothed_sequence)
        else:
            smoothed_sequence = self._moving_average_smooth(smoothed_sequence)

        # Convert back to list
        smoothed_joints = [smoothed_sequence[t] for t in range(T)]

        return smoothed_joints, bad_frames

    def _detect_bad_frames(self,
                           joints_sequence: np.ndarray,
                           confidence_sequence: np.ndarray) -> List[bool]:
        """
        Detect frames with bad detections

        Criteria:
        1. Large displacement from previous frame
        2. Low confidence on multiple joints
        3. Anatomically impossible poses
        """
        T, J, _ = joints_sequence.shape
        bad_frames = [False] * T

        for t in range(1, T):
            # Criterion 1: Large displacement
            displacement = np.linalg.norm(
                joints_sequence[t] - joints_sequence[t-1],
                axis=1
            )  # [24]

            # Check if any joint moved too much
            if np.any(displacement > self.max_displacement_px):
                bad_frames[t] = True
                continue

            # Criterion 2: Low confidence
            low_conf_joints = confidence_sequence[t] < self.min_confidence

            # Key body joints (torso + legs)
            key_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Pelvis to feet
            key_low_conf = np.sum(low_conf_joints[key_joints])

            # If >30% of key joints have low confidence
            if key_low_conf > len(key_joints) * 0.3:
                bad_frames[t] = True
                continue

            # Criterion 3: Anatomically impossible (optional check)
            # e.g., knees above hips, feet above knees, etc.
            if self._is_anatomically_impossible(joints_sequence[t]):
                bad_frames[t] = True

        return bad_frames

    def _is_anatomically_impossible(self, joints_2d: np.ndarray) -> bool:
        """
        Check for anatomically impossible 2D poses

        Args:
            joints_2d: [24, 2] joint positions

        Returns:
            True if pose is impossible
        """
        # Simple checks on Y-coordinates (assuming Y increases downward)
        pelvis_y = joints_2d[0, 1]
        l_knee_y = joints_2d[4, 1]
        r_knee_y = joints_2d[5, 1]
        l_ankle_y = joints_2d[7, 1]
        r_ankle_y = joints_2d[8, 1]

        # Knees should be below pelvis (in most cases)
        # Allow some tolerance for sitting/lying poses
        if l_knee_y < pelvis_y - 200 or r_knee_y < pelvis_y - 200:
            return True

        # Ankles should be below knees (in most cases)
        if l_ankle_y < l_knee_y - 100 or r_ankle_y < r_knee_y - 100:
            return True

        return False

    def _interpolate_bad_frames(self,
                                joints_sequence: np.ndarray,
                                bad_frames: List[bool]) -> np.ndarray:
        """
        Interpolate bad frames using neighboring good frames

        Uses linear interpolation between nearest good frames
        """
        T, J, _ = joints_sequence.shape
        smoothed = joints_sequence.copy()

        for t in range(T):
            if not bad_frames[t]:
                continue

            # Find previous good frame
            prev_good = None
            for i in range(t-1, -1, -1):
                if not bad_frames[i]:
                    prev_good = i
                    break

            # Find next good frame
            next_good = None
            for i in range(t+1, T):
                if not bad_frames[i]:
                    next_good = i
                    break

            # Interpolate
            if prev_good is not None and next_good is not None:
                # Linear interpolation
                alpha = (t - prev_good) / (next_good - prev_good)
                smoothed[t] = (1 - alpha) * joints_sequence[prev_good] + \
                              alpha * joints_sequence[next_good]
            elif prev_good is not None:
                # Forward fill
                smoothed[t] = joints_sequence[prev_good]
            elif next_good is not None:
                # Backward fill
                smoothed[t] = joints_sequence[next_good]
            # else: keep original (all frames are bad)

        return smoothed

    def _moving_average_smooth(self, joints_sequence: np.ndarray) -> np.ndarray:
        """
        Apply moving average smoothing across time

        Args:
            joints_sequence: [T, 24, 2]

        Returns:
            Smoothed sequence [T, 24, 2]
        """
        T, J, _ = joints_sequence.shape
        window = self.smoothing_window

        if T < window:
            return joints_sequence  # Too short to smooth

        smoothed = np.zeros_like(joints_sequence)

        for t in range(T):
            # Define window
            start = max(0, t - window // 2)
            end = min(T, t + window // 2 + 1)

            # Average over window
            smoothed[t] = np.mean(joints_sequence[start:end], axis=0)

        return smoothed

    def _kalman_smooth(self, joints_sequence: np.ndarray) -> np.ndarray:
        """
        Apply Kalman smoothing (more advanced)

        Kalman filter for each joint independently
        """
        T, J, D = joints_sequence.shape
        smoothed = np.zeros_like(joints_sequence)

        # Simple Kalman parameters
        process_noise = 0.1  # How much we expect joints to move
        measurement_noise = 5.0  # Detector noise in pixels

        for j in range(J):
            for d in range(D):  # X and Y separately
                # Initialize
                x = joints_sequence[0, j, d]  # Initial state
                P = 100.0  # Initial uncertainty

                trajectory = []

                for t in range(T):
                    # Predict (assume constant position)
                    x_pred = x
                    P_pred = P + process_noise

                    # Update with measurement
                    measurement = joints_sequence[t, j, d]
                    K = P_pred / (P_pred + measurement_noise)  # Kalman gain
                    x = x_pred + K * (measurement - x_pred)
                    P = (1 - K) * P_pred

                    trajectory.append(x)

                smoothed[:, j, d] = trajectory

        return smoothed


def smooth_detections_simple(joints_2d_list: List[np.ndarray],
                             confidence_list: List[np.ndarray],
                             max_jump: float = 100.0,
                             min_conf: float = 0.2) -> List[np.ndarray]:
    """
    Simple convenience function for detection smoothing

    Args:
        joints_2d_list: List of [24, 2] arrays
        confidence_list: List of [24] confidence arrays
        max_jump: Maximum allowed jump between frames (pixels)
        min_conf: Minimum confidence threshold

    Returns:
        Smoothed detections
    """
    smoother = DetectionSmoother(
        max_displacement_px=max_jump,
        min_confidence=min_conf,
        use_kalman=False,
        smoothing_window=5
    )

    smoothed, bad_frames = smoother.smooth_detections(joints_2d_list, confidence_list)

    num_bad = sum(bad_frames)
    if num_bad > 0:
        print(f"Smoothed {num_bad}/{len(bad_frames)} frames with bad detections")

    return smoothed


# Example usage:
if __name__ == "__main__":
    # Simulate detections with outliers
    T = 100
    joints_2d_list = []
    confidence_list = []

    for t in range(T):
        joints = np.random.randn(24, 2) * 10 + np.array([500, 500])
        confidence = np.random.rand(24) * 0.5 + 0.5

        # Add outlier at frame 50
        if t == 50:
            joints[5] += np.array([500, 500])  # Huge jump
            confidence[5] = 0.1

        joints_2d_list.append(joints)
        confidence_list.append(confidence)

    # Smooth
    smoother = DetectionSmoother(max_displacement_px=100, min_confidence=0.3)
    smoothed, bad_frames = smoother.smooth_detections(joints_2d_list, confidence_list)

    print(f"Detected {sum(bad_frames)} bad frames")
    print(f"Bad frame indices: {[i for i, b in enumerate(bad_frames) if b]}")
