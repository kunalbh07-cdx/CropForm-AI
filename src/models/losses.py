import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        # Convert model output into probabilities
        predictions = torch.sigmoid(logits)

        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Calculate intersection
        intersection = (predictions * targets).sum()

        # Calculate Dice coefficient
        dice = (
            (2.0 * intersection + self.smooth)
            /
            (
                predictions.sum()
                + targets.sum()
                + self.smooth
            )
        )

        # Dice Loss
        return 1.0 - dice


class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return bce_loss + dice_loss