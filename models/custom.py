import torch


class MNIST_Net(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 100)
        self.fc2 = torch.nn.Linear(100, n_classes)
        self.sig = torch.nn.Sigmoid()


    def emb_dim(self):
        return 100

    def forward(self, x, use_linear=False):
        x = torch.flatten(x, 1)
        e = self.sig(self.fc1(x))
        out = self.fc2(e)

        if use_linear:
            return out, e
        else:
            return out


class CIFAR10_Net(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(32*32*3, 100)
        self.fc2 = torch.nn.Linear(100, n_classes)
        self.sig = torch.nn.Sigmoid()


    def emb_dim(self):
        return 100

    def forward(self, x, use_linear=False):
        x = torch.flatten(x, 1)
        e = self.sig(self.fc1(x))
        out = self.fc2(e)

        if use_linear:
            return out, e
        else:
            return out
