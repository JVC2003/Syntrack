from dataset import AudioMatchingDataset
from torch.utils.data import DataLoader
from utils import collate_fn

import torch
from utils import loss_fn

ts_pred = torch.tensor([0.52])
conf_pred = torch.tensor([0.6])
ts_true = torch.tensor([0.5])

loss, details = loss_fn(ts_pred, conf_pred, ts_true)
print("Loss total:", loss.item())
print("Detalles:", details)
