from typing import List, NamedTuple

import torch
from torchaudio.models.decoder import ctc_decoder

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.vocab = vocab

        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        decoded_tokens = []
        current_token = None

        for index in inds:
            token = self.ind2char[index]
            if current_token is None or current_token != token:
                current_token = token
                if token != self.EMPTY_TOK:
                    decoded_tokens.append(token)
        return ''.join(decoded_tokens)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100, nbest: int = 1) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        decoder = ctc_decoder(
            lexicon=None,
            tokens=self.vocab,
            lm=None,
            nbest=nbest,
            beam_size=beam_size,
            blank_token=' ',
            sil_token=CTCCharTextEncoder.EMPTY_TOK
        )
        ctc_hypos = decoder(emissions=probs.unsqueeze(dim=0))
        hypos = []
        for hypo in ctc_hypos[0]:
            text = self.ctc_decode(hypo.tokens.tolist())
            hypos.append(Hypothesis(text = text,
                                    prob = hypo.score))
        return hypos
