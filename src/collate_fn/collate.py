import logging
from typing import List

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
import numpy as np


def collate_fn(dataset_items: List[dict]):
    # Strongly inspired by seminar code
    lengths = np.array([d["text"].shape[0] for d in dataset_items])
    sorted_index = np.argsort(-lengths)

    texts = [dataset_items[index]["text"] for index in sorted_index]
    mel_targets = [dataset_items[index]["mel_target"] for index in sorted_index]
    durations = [dataset_items[index]["duration"] for index in sorted_index]
    pitches = [dataset_items[index]["pitch"] for index in sorted_index]
    energies = [dataset_items[index]["energy"] for index in sorted_index]

    text_length = np.zeros(shape=(len(texts),))
    for i, text in enumerate(texts):
        text_length[i] = text.shape[0]

    src_positions = list()
    max_length = int(max(text_length))
    for length_src_row in text_length:
        src_positions.append(
            np.pad(
                [i + 1 for i in range(int(length_src_row))],
                (0, max_len - int(length_src_row)),
                "constant",
            )
        )
    src_positions = torch.from_numpy(np.array(src_positions))

    mel_length = np.zeros(shape=(len(mel_targets),))
    for i, mel in enumerate(mel_targets):
        mel_length[i] = mel.shape[0]

    mel_pos = list()
    max_mel_length = int(max(mel_length))
    for length_mel_row in mel_length:
        mel_pos.append(
            np.pad(
                [i + 1 for i in range(int(length_mel_row))],
                (0, max_mel_length - int(length_mel_row)),
                "constant",
            )
        )
    mel_pos = torch.from_numpy(np.array(mel_pos))

    src_seq = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations, max_pad_len=max_len)
    pitches = pad_1D_tensor(pitches)
    energies = pad_1D_tensor(energies)
    mel_targets = pad_2D_tensor(mel_targets)
    src_seq = pad_1D_tensor(texts, max_pad_len=max_len)

    out = {
        "src_sequence": src_seq,
        "src_positions": src_pos,
        "mel_target": mel_targets,
        "duration_target": durations,
        "pitch_target": pitches,
        "energy_target": energies,
        "mel_positions": mel_pos,
        "mel_max_length": max_mel_len,
    }

    return out


def pad_1D_tensor(inputs, PAD=0, max_pad_len=None):
    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    if max_pad_len:
        max_len = max(max_len, max_pad_len)
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D_tensor(inputs, maxlen=None):
    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len - x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output
