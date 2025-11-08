
import torch, math, logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def build_pap_texts(U:int, H:int)->List[str]:
    """
    Build domain, instruction, and statistics templates.
    Stats text will be supplied at runtime as an additional prompt.
    """
    domain = ("Task: Predict future optimal beams (continuous AoD in radians) "
              "for mmWave downlink with a ULA at the BS from past observations.")
    instr  = (f"Given the past U={U} steps of normalized beam indices q_a and AoDs, "
              f"output the next H={H} steps of AoDs.")
    # statistics placeholder token; actual stats are concatenated later
    stats  = "Stats: trend, autocorrelation lags."
    return [domain, instr, stats]

def safe_load_gpt2():
    """
    Try to load GPT-2 tokenizer+model. If unavailable (offline), return None.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("gpt2")
        mdl = AutoModel.from_pretrained("gpt2")
        return tok, mdl
    except Exception as e:
        logger.warning("Falling back: GPT-2 not available (%s)", e)
        return None, None

def embed_texts(texts:List[str], tokenizer, model, device):
    """
    Convert a list of texts to a single embedding tensor [T, D].
    Use last hidden states averaged over tokens.
    """
    with torch.no_grad():
        outs = []
        for s in texts:
            tok = tokenizer(s, return_tensors="pt").to(device)
            h = model(**tok).last_hidden_state.mean(dim=1)  # [1,D]
            outs.append(h)
        return torch.cat(outs, dim=0)  # [T,D]

class PaPPrototypes(torch.nn.Module):
    """
    A small learnable subset of token-like vectors used to reprogram patches into GPT-2 space.
    """
    def __init__(self, vocab_small:int, gpt2_dim:int):
        super().__init__()
        self.proto = torch.nn.Parameter(torch.randn(vocab_small, gpt2_dim) / math.sqrt(gpt2_dim))

    def forward(self, Bx:torch.Tensor):
        """
        Optionally could pick nearest prototypes; here we just return them for attention K/V.
        """
        return self.proto  # [V', D]
