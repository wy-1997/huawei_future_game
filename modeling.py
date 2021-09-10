import torch
import torch.nn as nn
import torch.nn.functional as F
from finetune_config import config
import numpy as np


batch_size = config["batch_size"]
dim = config["output_dim"]
K = config["neg_num"]



# infoNCE loss
class infoNCE(nn.Module):
    def __init__(self):
        super(infoNCE, self).__init__()
        self.T = 0.07
        self.cross_entropy = nn.CrossEntropyLoss().cuda()

    def forward(self, query, pos, neg):
        query = F.normalize(query.view(batch_size, 1, dim), p=2, dim=2)
        pos = F.normalize(pos.view(batch_size, 1, dim), p=2, dim=2)
        neg = F.normalize(neg.view(batch_size, K, dim), p=2, dim=2)

        pos_score = torch.bmm(query, pos.transpose(1, 2)) # B*1*1
        neg_score = torch.bmm(query, neg.transpose(1, 2))  # B*1*K

        # logits:B*(K+1)
        logits = torch.cat([pos_score, neg_score], dim=2).squeeze()
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        info_loss = self.cross_entropy(logits, labels)

        return info_loss

class MLP(nn.Module):
    def __init__(self, in_dim):
        super(MLP, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )

    def forward(self, x):
        x = self.projection(x)
        return x
