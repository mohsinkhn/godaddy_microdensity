from torch import nn


class SmapeLoss(nn.Module):
    def __init__(self):
        super(SmapeLoss, self).__init__()

    def forward(self, output, target):
        loss = 2 * (output - target).abs() / (output.abs() + target.abs() + 1e-6) * 100
        return loss.mean()
