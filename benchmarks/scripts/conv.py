import torch

class TimeConv:
    def setup(self):
        torch.set_num_threads(1)
        self.conv = torch.nn.Conv2d(1, 1, 2)
        self.x = torch.ones((1, 1, 4, 4))

    def time_forward(self):
        self.conv(self.x)

