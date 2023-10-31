import logging
from typing import List

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F

def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}

    values_to_pad = ['spectrogram', 'text_encoded', 'audio']

    for value in values_to_pad:
        lengths = [item[value].shape[-1] for item in dataset_items]
        result_batch[f'{value}_length'] = torch.tensor(lengths)

        size_to_pad = max(lengths)
        result_batch[value] = torch.cat([
            F.pad(
                input = item[value], 
                pad = (0, size_to_pad - item[value].shape[-1]),
                value = 0
            ) 
            for item in dataset_items
        ])

    result_batch['text'] = []
    result_batch['audio_path'] = []

    for item in dataset_items:
        result_batch['text'].append(item['text'])
        result_batch['audio_path'].append(item['audio_path'])

    return result_batch