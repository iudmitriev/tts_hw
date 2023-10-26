from typing import List, NamedTuple
from string import ascii_lowercase
from collections import defaultdict

import torch
from torchaudio.models.decoder import download_pretrained_files
from pyctcdecode import build_ctcdecoder


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
        hypos: List[Hypothesis] = [Hypothesis("", 0)]

        for frame in probs:
            hypos = self._extend_hypos(frame, hypos)
            hypos = sorted(hypos, key=lambda x: x.prob, reverse=True)[:beam_size]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
    

    def _extend_hypos(self, frame, hypos):
        new_hypos = defaultdict(float)
        for token_index, token_proba in enumerate(frame):
            token = self.ind2char[token_index]
            for hypo in hypos:
                current_token = hypo.text[-1] if len(hypo.text) != 0 else ""
                
                if token == current_token or token == self.EMPTY_TOK:
                    new_text = hypo.text
                else:
                    new_text = hypo.text + token
                new_prob = hypo.prob * token_proba
                new_hypos[new_text] += new_prob
        hypos = [Hypothesis(text, prob) for text, prob in new_hypos.items()]
        return hypos


class CTCCharTextEncoderWithLM(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, model_path=None, alpha: float = 0.5, beta: float = 1.0):
        if alphabet is None:
            alphabet = [''] + list(ascii_lowercase) + [' ']
        super().__init__(alphabet)

        self.ind2char = dict(enumerate(alphabet))

        if model_path is None:
            files = download_pretrained_files("librispeech-4-gram")
            model_path = files.lm

        self.decoder = build_ctcdecoder(
            alphabet,
            kenlm_model_path=model_path,
            alpha=alpha,
            beta=beta,
        )
    
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

    def ctc_decode_with_lm(self, logits: torch.tensor) -> str:
        return self.decoder.decode(logits.detach().numpy())
