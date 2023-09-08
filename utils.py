import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast
from torch import Tensor
import numpy as np
import torch
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class GEGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def geglu(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x: Tensor) -> Tensor:
        return self.geglu(x)
    
class ReGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def reglu(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)

    def forward(self, x: Tensor) -> Tensor:
        return self.reglu(x)
    
def get_keyphrase(input_ids, output, tokenizer=None):
    if not tokenizer: tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    bs = len(input_ids)
    scores, pred = output.max(dim=-1)
    scores, pred = scores.exp().cpu().tolist(), pred.cpu().tolist()
    input_ids = input_ids.cpu().tolist()
    ret = []
    for b in range(bs): 
        keyphrases = []
        cur_keyphrase = []
        cur_scores = []
        for token, p, score in zip(input_ids[b], pred[b], scores[b]):
            if p == 1:  # part of a keyphrase
                cur_keyphrase.append(token)
                cur_scores.append(score)
            elif cur_keyphrase:
                keyphrase = "".join(tokenizer.decode(cur_keyphrase))
                avg_score = sum(cur_scores) / len(cur_scores)
                keyphrases.append((keyphrase, avg_score))
                cur_keyphrase = []
                cur_scores = []

        # don't forget the last keyphrase if the text ends with one
        if cur_keyphrase:
            keyphrase = " ".join(tokenizer.decode(cur_keyphrase))
            avg_score = sum(cur_scores) / len(cur_scores)
            keyphrases.append((keyphrase, avg_score))

        # sort by score and return the top keyphrase
        keyphrases.sort(key=lambda x: x[1], reverse=True)
        kp = keyphrases[0][0] if keyphrases else ""
        ret.append(kp)
    return ret