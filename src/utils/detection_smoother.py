import numpy as np
from typing import List, Tuple, Optional


class DetectionSmoother:

    def __init__(self,
                 max_displacement_px: float = 100.0,
                 min_confidence: float = 0.2,
                 use_kalman: bool = False,
                 smoothing_window: int = 5):
        self.max_displacement_px = max_displacement_px
        self.min_confidence = min_confidence
        self.use_kalman = use_kalman
        self.smoothing_window = smoothing_window

    def smooth_detections(self,
                          joints_2d_list: List[np.ndarray],
                          confidence_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[bool]]:

        if len(joints_2d_list) == 0:
            return [], []

        joints_sequence = np.array(joints_2d_list)
        confidence_sequence = np.array(confidence_list)

        T, J, _ = joints_sequence.shape

        bad_frames = self._detect_bad_frames(joints_sequence, confidence_sequence)

        smoothed_sequence = self._interpolate_bad_frames(joints_sequence, bad_frames)

        if self.use_kalman:
            smoothed_sequence = self._kalman_smooth(smoothed_sequence)
        else:
            smoothed_sequence = self._moving_average_smooth(smoothed_sequence)

        smoothed_joints = [smoothed_sequence[t] for t in range(T)]

        return smoothed_joints, bad_frames

    def _detect_bad_frames(self,
                           joints_sequence: np.ndarray,
                           confidence_sequence: np.ndarray) -> List[bool]:
        T, J, _ = joints_sequence.shape
        bad_frames = [False] * T

        for t in range(1, T):
            displacement = np.linalg.norm(
                joints_sequence[t] - joints_sequence[t-1],
                axis=1
            )

            if np.any(displacement > self.max_displacement_px):
                bad_frames[t] = True
                continue

            low_conf_joints = confidence_sequence[t] < self.min_confidence

            key_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            key_low_conf = np.sum(low_conf_joints[key_joints])

            if key_low_conf > len(key_joints) * 0.3:
                bad_frames[t] = True
                continue

            if self._is_anatomically_impossible(joints_sequence[t]):
                bad_frames[t] = True

        return bad_frames

    def _is_anatomically_impossible(self, joints_2d: np.ndarray) -> bool:
        pelvis_y = joints_2d[0, 1]
        l_knee_y = joints_2d[4, 1]
        r_knee_y = joints_2d[5, 1]
        l_ankle_y = joints_2d[7, 1]
        r_ankle_y = joints_2d[8, 1]

        if l_knee_y < pelvis_y - 200 or r_knee_y < pelvis_y - 200:
            return True

        if l_ankle_y < l_knee_y - 100 or r_ankle_y < r_knee_y - 100:
            return True

        return False

    def _interpolate_bad_frames(self,
                                joints_sequence: np.ndarray,
                                bad_frames: List[bool]) -> np.ndarray:
        T, J, _ = joints_sequence.shape
        smoothed = joints_sequence.copy()

        for t in range(T):
            if not bad_frames[t]:
                continue

            prev_good = None
            for i in range(t-1, -1, -1):
                if not bad_frames[i]:
                    prev_good = i
                    break

            next_good = None
            for i in range(t+1, T):
                if not bad_frames[i]:
                    next_good = i
                    break

            if prev_good is not None and next_good is not None:
                alpha = (t - prev_good) / (next_good - prev_good)
                smoothed[t] = (1 - alpha) * joints_sequence[prev_good] + \
                              alpha * joints_sequence[next_good]
            elif prev_good is not None:
                smoothed[t] = joints_sequence[prev_good]
            elif next_good is not None:
                smoothed[t] = joints_sequence[next_good]

        return smoothed

    def _moving_average_smooth(self, joints_sequence: np.ndarray) -> np.ndarray:
        T, J, _ = joints_sequence.shape
        window = self.smoothing_window

        if T < window:
            return joints_sequence

        smoothed = np.zeros_like(joints_sequence)

        for t in range(T):
            start = max(0, t - window // 2)
            end = min(T, t + window // 2 + 1)

            smoothed[t] = np.mean(joints_sequence[start:end], axis=0)

        return smoothed

    def _kalman_smooth(self, joints_sequence: np.ndarray) -> np.ndarray:
        T, J, D = joints_sequence.shape
        smoothed = np.zeros_like(joints_sequence)

        process_noise = 0.1
        measurement_noise = 5.0

        for j in range(J):
            for d in range(D):
                x = joints_sequence[0, j, d]
                P = 100.0

                trajectory = []

                for t in range(T):
                    x_pred = x
                    P_pred = P + process_noise

                    measurement = joints_sequence[t, j, d]
                    K = P_pred / (P_pred + measurement_noise)
                    x = x_pred + K * (measurement - x_pred)
                    P = (1 - K) * P_pred

                    trajectory.append(x)

                smoothed[:, j, d] = trajectory

        return smoothed


def smooth_detections_simple(joints_2d_list: List[np.ndarray],
                             confidence_list: List[np.ndarray],
                             max_jump: float = 100.0,
                             min_conf: float = 0.2) -> List[np.ndarray]:
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
