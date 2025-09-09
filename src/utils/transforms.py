import torch

class NormalizerJoints2d:
    def __init__(self):
        # No fixed img_size anymore - we use resolution from each sample
        pass
        
    def __call__(self, sample):
        joints_2d = sample["joints_2d"].clone()
        
        # Get resolution from sample [width, height]
        resolution = sample["resolution"]
        
        if isinstance(resolution, torch.Tensor):
            width = resolution[0].item()
            height = resolution[1].item()
        else:
            width = resolution[0]
            height = resolution[1]
        
        # Normalize based on actual resolution
        # X coordinates normalized by width to [-1, 1]
        joints_2d[:, 0] = (joints_2d[:, 0] / width) * 2 - 1
        # Y coordinates normalized by height to [-1, 1]
        joints_2d[:, 1] = (joints_2d[:, 1] / height) * 2 - 1
        
        sample["joints_2d"] = joints_2d
        
        return sample