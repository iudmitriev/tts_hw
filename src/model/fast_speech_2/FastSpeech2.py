import torch
from torch import nn
import torch.nn.functional as F

def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech2Model(nn.Module):
    def __init__(self, encoder, variance_adapter, decoder, num_mels):
        super().__init__()
        self.encoder = encoder
        self.variance_adaptor = variance_adapter
        self.decoder = decoder
        self.mel_linear = nn.Linear(
            decoder.hidden_dim,
            num_mels,
        )


    def forward(
        self,
        src_sequence,
        src_positions,
        mel_positions=None,
        mel_max_length=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        duration_control = 1.0,
        pitch_control = 1.0,
        energy_control = 1.0,
        **kwargs
    ):
        mel_prediction, _ = self.encoder(src_sequence, src_positions)
        if self.training:
            predictions = self.variance_adaptor(
                mel_prediction = mel_prediction,
                mel_max_length=mel_max_length,
                duration_target=duration_target,
                pitch_target=pitch_target,
                energy_target=energy_target,
                duration_control=duration_control,
                pitch_control=pitch_control,
                energy_control=energy_control
            )
            mel_prediction = predictions['mel_prediction']
            mel_prediction = self.decoder(mel_prediction, mel_positions)
            mel_prediction = self.zero_beyond_length(mel_prediction, mel_positions, mel_max_length)
            mel_prediction = self.mel_linear(mel_prediction)
        else:
            predictions = self.variance_adaptor(mel_prediction)            
            mel_prediction = predictions['mel_prediction']
            mel_positions = predictions['mel_positions']
            mel_prediction = self.decoder(mel_prediction, mel_positions)
            mel_prediction = self.mel_linear(mel_prediction)

        return {"mel_prediction": mel_prediction, 
                "log_duration_prediction": predictions['log_duration_prediction'],
                "pitch_prediction": predictions['pitch_prediction'],
                "energy_prediction": predictions['energy_prediction']}


    @staticmethod
    def zero_beyond_length(mel_prediction, mel_positions, mel_max_length):
        lengths = torch.max(mel_positions, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_prediction.size(-1))
        return mel_prediction.masked_fill(mask, 0.)
