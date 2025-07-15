# utils.py
import torch
import torch.nn as nn
def collate_fn(batch):
    clips = [item['clip'] for item in batch]
    songs = [item['song'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    clip_ids = [item['clip_id'] for item in batch]
    valid = [item['valid'] for item in batch]

    return {
        'clip': clips,
        'song': songs,
        'target': targets,
        'clip_id': clip_ids,
        'valid': valid,
    }

def loss_fn(ts_pred, conf_pred, ts_true, delta=0.05):
    ts_loss = nn.functional.smooth_l1_loss(ts_pred, ts_true)
    success = (torch.abs(ts_pred - ts_true) < delta).float()
    conf_loss = nn.functional.binary_cross_entropy(conf_pred, success)
    return ts_loss + conf_loss, {'ts_loss': ts_loss.item(), 'conf_loss': conf_loss.item()}