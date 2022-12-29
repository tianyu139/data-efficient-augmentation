import torchvision
import torch
import threading
from collections import defaultdict


class ResnetWrapper(torch.nn.Module):
    def __init__(self, resnet):
        super(ResnetWrapper, self).__init__()
        self.model = resnet
        self.ed = self.model.state_dict()['fc.weight'].shape[-1]
        self.target_outputs = defaultdict(lambda: None)
        self.model.fc.register_forward_hook(self.get_linear)

    def emb_dim(self):
        return self.ed

    def get_linear(self, _, i, o):
        self.target_outputs[threading.get_native_id()] = i[0]

    def forward(self, x, use_linear=False):
        out = self.model(x)
        if not use_linear:
            return out
        else:
            thread_id = threading.get_native_id()
            e = self.target_outputs[thread_id].clone()
            del self.target_outputs[thread_id]

            return out, e


