import torch
import torch.nn as nn

class customLoss(nn.Module):
    def __init__(self, weight):
        # --------------------------------------------
        # Initialization
        # --------------------------------------------
        super(customLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.weight = weight

    def forward(self, outputs, targets):
        # --------------------------------------------
        # Define forward pass
        # --------------------------------------------
        # Transform targets to one-hot vector
        targets_onehot = torch.zeros_like(outputs)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets.unsqueeze(-1), 1)

        # nn.CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss()
        outputs = self.softmax(outputs)
        self.weight = self.weight.expand_as(outputs)
        loss = -targets_onehot.float() * torch.log(outputs)
        return torch.mean(self.weight * loss)