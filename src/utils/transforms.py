import torch

class NormalizerJoints2d:
    def __init__(self):
        pass

    def __call__(self, sample):
        joints_2d = sample["joints_2d"].clone()

        resolution = sample["resolution"]

        if isinstance(resolution, torch.Tensor):
            width = resolution[0].item()
            height = resolution[1].item()
        else:
            width = resolution[0]
            height = resolution[1]

        joints_2d[:, 0] = (joints_2d[:, 0] / width) * 2 - 1
        joints_2d[:, 1] = (joints_2d[:, 1] / height) * 2 - 1

        sample["joints_2d"] = joints_2d

        return sample
