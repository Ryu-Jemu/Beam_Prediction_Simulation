import torch, torch.nn as nn, math
from setting import Config

class PatchEmbed(nn.Module):
    def __init__(self, c_in:int, d_model:int, k:int, s:int):
        super().__init__()
        self.proj = nn.Conv1d(c_in, d_model, kernel_size=k, stride=s)
    def forward(self, x):  # x:[B,C,U]
        z = self.proj(x)          # [B,Dm,P]
        return z.transpose(1,2)   # [B,P,Dm]

class Encoder(nn.Module):
    def __init__(self, d_model:int, n_heads:int, n_layers:int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
    def forward(self, x):  # [B,P,Dm]
        return self.enc(x)

class PaPGPT2Regressor(nn.Module):
    """
    Input:  X [B,C,U]  (features over past U)
    Output: residual (sin,cos) [B,H,2]
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        c_in = cfg.feature_dim + (2 if cfg.include_aod_in_features else 0)
        self.patch = PatchEmbed(c_in, cfg.d_model, cfg.patch_len, cfg.patch_stride)
        self.enc   = Encoder(cfg.d_model, cfg.n_heads, cfg.n_layers)
        self.head  = nn.Linear(cfg.d_model, 2)
        self.time_proj = nn.Linear(cfg.d_model, cfg.H*2)  # fallback if P < H
        self.act   = nn.Tanh()

        # optional text branch
        self.use_text = bool(getattr(cfg, "use_gpt2", False))
        if self.use_text:
            try:
                from transformers import AutoTokenizer, AutoModel
                self.tok = AutoTokenizer.from_pretrained("gpt2")
                self.gpt2 = AutoModel.from_pretrained("gpt2")
                self.text_proj = nn.Linear(self.gpt2.config.n_embd, cfg.d_model)
            except Exception:
                self.use_text = False

    def forward(self, x, stats_text:str=None):  # x:[B,C,U]
        B = x.size(0)
        z = self.patch(x)                # [B,P,Dm]
        z = self.enc(z)                  # [B,P,Dm]

        if self.use_text and stats_text is not None:
            dev = next(self.parameters()).device
            inputs = self.tok([str(stats_text)]*B, return_tensors="pt", padding=True, truncation=True)
            htxt = self.gpt2(**inputs).last_hidden_state.mean(dim=1)  # CPU by default
            htxt = self.text_proj(htxt.to(dev))                       # project to Dm on device
            z = z + htxt.unsqueeze(1)                                 # broadcast

        P = z.size(1)
        if P >= self.cfg.H:
            zH = z[:, -self.cfg.H:, :]      # last H patches
            out = self.head(zH)             # [B,H,2]
        else:
            zg = torch.mean(z, dim=1)       # [B,Dm]
            out = self.time_proj(zg).view(B, self.cfg.H, 2)

        return self.act(out)                 # residual in [-1,1]