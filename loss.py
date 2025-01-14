import torch
import torch.nn as nn


# Contrastive Loss Function Definition
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings_left, embeddings_right, labels):
        # Calculate the Euclidean distance between the two embeddings
        euclidean_distance = nn.functional.pairwise_distance(
            embeddings_left,
            embeddings_right,
        )

        # Contrastive loss formula
        loss_positive = labels * torch.pow(euclidean_distance, 2)
        loss_negative = (1 - labels) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0),
            2,
        )

        # Sum the losses
        loss = torch.mean(loss_positive + loss_negative)
        return loss
