# SMPL skeleton configuration
# 24 joints: Pelvis(0) to R_Hand(23)
SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1",
    "L_Knee", "R_Knee", "Spine2", 
    "L_Ankle", "R_Ankle", "Spine3",
    "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", 
    "L_Wrist", "R_Wrist",
    "L_Hand", "R_Hand"
]

# SMPL skeleton connections (23 edges)
SMPL_SKELETON = [
    # Left leg chain: Pelvis -> L_Hip -> L_Knee -> L_Ankle -> L_Foot
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right leg chain: Pelvis -> R_Hip -> R_Knee -> R_Ankle -> R_Foot  
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Spine chain: Pelvis -> Spine1 -> Spine2 -> Spine3 -> Neck -> Head
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left arm chain: Spine3 -> L_Collar -> L_Shoulder -> L_Elbow -> L_Wrist -> L_Hand
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    # Right arm chain: Spine3 -> R_Collar -> R_Shoulder -> R_Elbow -> R_Wrist -> R_Hand
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23)
]