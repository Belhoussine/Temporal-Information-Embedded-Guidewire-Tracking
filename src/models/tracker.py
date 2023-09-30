import torch
import timm
import einops
import tqdm


class CoTracker():
    def __init__(self):
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker_w8")
        if torch.cuda.is_available():
            self.model.cuda()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)