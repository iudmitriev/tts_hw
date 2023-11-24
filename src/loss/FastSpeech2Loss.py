import torch
from torch import nn
import torch.nn.functional as F

class FastSpeech2Loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        mel_prediction,
        pitch_prediction,
        energy_prediction,
        log_duration_prediction,
        mel_target,
        pitch_target,
        energy_target,
        duration_target,
        **kwargs,
    ):
        log_duration_targets = torch.log(duration_target.float() + 1)
        log_pitch_targets = torch.log(pitch_target.float() + 1)
        log_energy_targets = torch.log(energy_target + 1)

        mel_loss = F.l1_loss(mel_prediction, mel_target)
        pitch_loss = F.mse_loss(pitch_prediction, log_pitch_targets)
        energy_loss = F.mse_loss(energy_prediction, log_energy_targets)
        duration_loss = F.mse_loss(log_duration_prediction, log_duration_targets)

        total_loss = mel_loss + duration_loss + pitch_loss + energy_loss
        return total_loss, mel_loss, pitch_loss, energy_loss, duration_loss
