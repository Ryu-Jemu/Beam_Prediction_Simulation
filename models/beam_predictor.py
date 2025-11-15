"""
Main beam prediction model with LLM integration
Architecture: Patch Embedding → Cross-Attention → Reprogramming → PaP → GPT-2 → Output
"""
import torch
import torch.nn as nn
import math
from typing import Optional

from config import Config
# Use absolute imports to avoid issues when this module is imported as part
# of a package.  The submodules live at the project root, so we import
# directly from them rather than using relative imports (which only work
# when this file is inside a package).
# Robustly import the pap_gpt2 submodule and the RevIN class.  The
# imports below first attempt to resolve relative modules when this file
# is part of a package (e.g., `models`).  If that fails, they fall back
# to absolute imports and finally to dynamically loading the modules
# from candidate file paths.  This makes the code resilient to
# different project structures.
try:
    # Try relative import when pap_gpt2.py is in the same package
    from .pap_gpt2 import (
        PatchReprogramming,
        PromptAsPrefix,
        GPT2Backbone,
        build_pap_prompts,
    )  # type: ignore
except Exception:
    try:
        # Fall back to absolute import
        from pap_gpt2 import (
            PatchReprogramming,
            PromptAsPrefix,
            GPT2Backbone,
            build_pap_prompts,
        )  # type: ignore
    except Exception:
        # Dynamically load pap_gpt2 from common locations
        import importlib.util as _importlib_util
        import os as _os
        import sys as _sys
        _module_dir = _os.path.dirname(__file__)
        _parent_dir = _os.path.abspath(_os.path.join(_module_dir, _os.pardir))
        _candidate_paths = [
            _os.path.join(_module_dir, 'pap_gpt2.py'),
            _os.path.join(_parent_dir, 'pap_gpt2.py'),
        ]
        _found = False
        for _pap_path in _candidate_paths:
            if _os.path.exists(_pap_path):
                _spec = _importlib_util.spec_from_file_location('pap_gpt2', _pap_path)
                if _spec and _spec.loader:
                    _pap_module = _importlib_util.module_from_spec(_spec)
                    _sys.modules['pap_gpt2'] = _pap_module
                    _spec.loader.exec_module(_pap_module)  # type: ignore[attr-defined]
                    PatchReprogramming = _pap_module.PatchReprogramming  # type: ignore[attr-defined]
                    PromptAsPrefix = _pap_module.PromptAsPrefix  # type: ignore[attr-defined]
                    GPT2Backbone = _pap_module.GPT2Backbone  # type: ignore[attr-defined]
                    build_pap_prompts = _pap_module.build_pap_prompts  # type: ignore[attr-defined]
                    _found = True
                    break
        if not _found:
            raise ImportError("Could not locate pap_gpt2 module in known locations")

# Import RevIN similarly
try:
    from .utils import RevIN  # type: ignore
except Exception:
    try:
        from utils import RevIN  # type: ignore
    except Exception:
        import importlib.util as _importlib_util2
        import os as _os2
        import sys as _sys2
        _module_dir2 = _os2.path.dirname(__file__)
        _parent_dir2 = _os2.path.abspath(_os2.path.join(_module_dir2, _os2.pardir))
        _candidate_paths2 = [
            _os2.path.join(_module_dir2, 'utils.py'),
            _os2.path.join(_parent_dir2, 'utils.py'),
            _os2.path.join(_parent_dir2, 'metrics.py'),
        ]
        _found2 = False
        for _util_path in _candidate_paths2:
            if _os2.path.exists(_util_path):
                _specu = _importlib_util2.spec_from_file_location('utils', _util_path)
                if _specu and _specu.loader:
                    _util_module = _importlib_util2.module_from_spec(_specu)
                    _sys2.modules['utils'] = _util_module
                    _specu.loader.exec_module(_util_module)  # type: ignore[attr-defined]
                    if hasattr(_util_module, 'RevIN'):
                        RevIN = _util_module.RevIN  # type: ignore[attr-defined]
                        _found2 = True
                        break
        if not _found2:
            raise ImportError("Could not locate RevIN in utils module")


class PatchEmbedding(nn.Module):
    """Convert time series to patch embeddings using Conv1D"""
    
    def __init__(
        self,
        c_in: int,
        d_model: int,
        patch_len: int,
        patch_stride: int
    ):
        """
        Args:
            c_in: number of input channels (features)
            d_model: embedding dimension
            patch_len: length of each patch
            patch_stride: stride for patches
        """
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        
        # Conv1D for patch embedding
        self.projection = nn.Conv1d(
            c_in,
            d_model,
            kernel_size=patch_len,
            stride=patch_stride
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, U] input time series
        
        Returns:
            patches: [B, P, d_model] patch embeddings
        """
        # Ensure contiguous for MPS compatibility
        if x.device.type == "mps":
            x = x.contiguous()
        
        z = self.projection(x)  # [B, d_model, P]
        z = z.transpose(1, 2)   # [B, P, d_model]
        
        # Ensure contiguous after transpose for MPS
        if z.device.type == "mps":
            z = z.contiguous()
        
        return z


class CrossVariableAttention(nn.Module):
    """Cross-variable attention to aggregate information from multiple features
    
    Uses a learnable query to aggregate from all input variables.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: model dimension
            n_heads: number of attention heads
            dropout: dropout rate
        """
        super().__init__()
        self.d_model = d_model
        
        # Learnable query for aggregation
        self.query = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregate across patches using attention
        
        Args:
            patch_embeddings: [B, P, d_model]
        
        Returns:
            aggregated: [B, P, d_model]
        """
        B, P, D = patch_embeddings.shape
        
        # Expand query for batch
        query = self.query.expand(B, P, -1)  # [B, P, d_model]
        
        # Self-attention with query
        attn_out, _ = self.attention(
            query,
            patch_embeddings,
            patch_embeddings
        )  # [B, P, d_model]
        
        # Residual and norm
        out = self.norm(patch_embeddings + attn_out)
        
        return out


class TransformerEncoder(nn.Module):
    """Transformer encoder for patch processing"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Args:
            d_model: model dimension
            n_heads: number of attention heads
            n_layers: number of layers
            dropout: dropout rate
            activation: activation function
        """
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
        
        Returns:
            encoded: [B, seq_len, d_model]
        """
        return self.encoder(x)


class OutputProjection(nn.Module):
    """Project hidden states to output predictions"""
    
    def __init__(
        self,
        d_model: int,
        H: int,
        output_dim: int = 2
    ):
        """
        Args:
            d_model: hidden dimension
            H: prediction horizon
            output_dim: output dimension (2 for sin, cos)
        """
        super().__init__()
        self.H = H
        self.output_dim = output_dim
        
        # Per-timestep projection
        self.proj = nn.Linear(d_model, output_dim)
        
        # Alternative: if not enough patches, use global projection
        self.global_proj = nn.Linear(d_model, H * output_dim)
        
        # Activation
        self.activation = nn.Tanh()  # Output in [-1, 1]
    
    def forward(self, hidden_states: torch.Tensor, num_prompts: int) -> torch.Tensor:
        """Project to output
        
        Args:
            hidden_states: [B, seq_len, d_model] from GPT-2/transformer
            num_prompts: number of prompt tokens to skip
        
        Returns:
            output: [B, H, 2] predictions (sin, cos)
        """
        B, seq_len, D = hidden_states.shape
        
        # Skip prompt tokens, use only patch outputs
        patch_states = hidden_states[:, num_prompts:, :]  # [B, P, D]
        
        P = patch_states.size(1)
        
        if P >= self.H:
            # Use last H patches
            last_states = patch_states[:, -self.H:, :]  # [B, H, D]
            output = self.proj(last_states)  # [B, H, 2]
        else:
            # Use global pooling + projection
            global_state = patch_states.mean(dim=1)  # [B, D]
            output = self.global_proj(global_state)  # [B, H*2]
            output = output.view(B, self.H, self.output_dim)  # [B, H, 2]
        
        # Apply activation
        output = self.activation(output)
        
        return output


class BeamPredictorLLM(nn.Module):
    """Complete beam prediction model with LLM
    
    Pipeline:
        1. Input normalization (RevIN)
        2. Patch embedding
        3. Cross-variable attention
        4. Patch reprogramming
        5. Prompt-as-prefix
        6. GPT-2 backbone (optional)
        7. Transformer encoder (if no GPT-2)
        8. Output projection
    """
    
    def __init__(self, cfg: Config):
        """
        Args:
            cfg: configuration
        """
        super().__init__()
        self.cfg = cfg
        
        # Input channels
        c_in = cfg.feature_dim
        
        # 1. RevIN for input normalization
        self.use_revin = cfg.use_revin
        if self.use_revin:
            self.revin = RevIN(c_in)
        
        # 2. Patch embedding
        self.patch_embedding = PatchEmbedding(
            c_in,
            cfg.d_model,
            cfg.patch_len,
            cfg.patch_stride
        )
        
        # 3. Cross-variable attention
        self.cross_var_attn = CrossVariableAttention(
            cfg.d_model,
            cfg.n_heads,
            cfg.dropout
        )
        
        # 4. GPT-2 integration (optional)
        self.use_gpt2 = cfg.use_gpt2
        
        if self.use_gpt2:
            try:
                # Patch reprogramming
                self.patch_reprogramming = PatchReprogramming(
                    cfg.d_model,
                    cfg.gpt2_dim,
                    cfg.text_prototype_vocab,
                    cfg.n_heads,
                    cfg.dropout
                )
                
                # GPT-2 backbone
                self.gpt2_backbone = GPT2Backbone(
                    cfg.gpt2_model,
                    cfg.gpt2_freeze
                )
                
                # Prompt-as-prefix
                self.pap = PromptAsPrefix(cfg.gpt2_dim)
                
                # Initialize prompts
                self._init_prompts()
                
                # Hidden dimension after GPT-2
                hidden_dim = cfg.gpt2_dim
                
            except Exception as e:
                print(f"Warning: Failed to initialize GPT-2: {e}")
                print("Falling back to Transformer-only mode")
                self.use_gpt2 = False
        
        if not self.use_gpt2:
            # Use Transformer encoder instead
            self.encoder = TransformerEncoder(
                cfg.d_model,
                cfg.n_heads,
                cfg.n_layers,
                cfg.dropout,
                cfg.activation
            )
            hidden_dim = cfg.d_model
        
        # 5. Output projection
        self.output_proj = OutputProjection(
            hidden_dim,
            cfg.H,
            output_dim=2
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _init_prompts(self):
        """Initialize prompt embeddings for PaP"""
        if not self.use_gpt2:
            return
        
        # Build prompt texts
        prompt_texts = build_pap_prompts(self.cfg.U, self.cfg.H)
        
        # Embed with GPT-2 (without statistics for now)
        device = next(self.parameters()).device
        prompt_embeddings = []
        
        for text in prompt_texts[:-1]:  # Exclude statistics template
            emb = self.gpt2_backbone.embed_text(text, device)  # [1, D]
            prompt_embeddings.append(emb)
        
        prompt_embeddings = torch.cat(prompt_embeddings, dim=0)  # [N, D]
        
        # Set in PaP module
        self.pap.set_prompt_embeddings(prompt_embeddings)
    
    def forward(
        self,
        x: torch.Tensor,
        stats_text: Optional[str] = None
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            x: [B, C, U] input features
            stats_text: optional statistics text for PaP
        
        Returns:
            predictions: [B, H, 2] residual (sin, cos)
        """
        B, C, U = x.shape
        
        # 1. Input normalization
        if self.use_revin:
            x = self.revin(x, mode="norm")  # [B, C, U]
        
        # 2. Patch embedding
        patches = self.patch_embedding(x)  # [B, P, d_model]
        
        # 3. Cross-variable attention
        patches = self.cross_var_attn(patches)  # [B, P, d_model]
        
        # 4-6. LLM processing
        if self.use_gpt2:
            # Reprogram patches to text space
            reprogrammed = self.patch_reprogramming(patches)  # [B, P, gpt2_dim]
            
            # Embed statistics text if provided
            if stats_text is not None and self.cfg.use_pap:
                device = reprogrammed.device
                stats_emb = self.gpt2_backbone.embed_text(stats_text, device)  # [1, D]
                stats_emb = stats_emb.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
            else:
                stats_emb = None
            
            # Apply PaP
            if self.cfg.use_pap:
                combined = self.pap(reprogrammed, stats_emb)  # [B, N+P(+1), gpt2_dim]
                num_prompts = self.pap.get_num_prompts()
                if stats_emb is not None:
                    num_prompts += 1
            else:
                combined = reprogrammed
                num_prompts = 0
            
            # Process through GPT-2
            hidden_states = self.gpt2_backbone(combined)  # [B, seq_len, gpt2_dim]
        
        else:
            # Use Transformer encoder
            hidden_states = self.encoder(patches)  # [B, P, d_model]
            num_prompts = 0
        
        # 7. Output projection
        output = self.output_proj(hidden_states, num_prompts)  # [B, H, 2]
        
        eps = 1e-8
        norm = torch.clamp(torch.linalg.norm(output, dim=-1, keepdim=True), min=eps)
        output = output / norm
        
        return output
    
    def predict_with_baseline(
        self,
        x: torch.Tensor,
        baseline_angles: torch.Tensor,
        stats_text: Optional[str] = None
    ) -> torch.Tensor:
        """Predict with CTRV baseline composition
        
        Args:
            x: [B, C, U] input features
            baseline_angles: [B, H] baseline AoD predictions
            stats_text: optional statistics text
        
        Returns:
            predictions: [B, H, 2] composed (sin, cos)
        """
        from utils import compose_residual_with_baseline
        
        # Get residual prediction
        residual = self.forward(x, stats_text)  # [B, H, 2]
        
        # Compose with baseline
        if self.cfg.use_ctrv_baseline and self.cfg.ctrv_weight > 0:
            # Weighted composition
            predictions = compose_residual_with_baseline(residual, baseline_angles)
            
            # Mix with direct prediction
            alpha = self.cfg.ctrv_weight
            predictions = alpha * predictions + (1 - alpha) * residual
        else:
            predictions = residual
        
        return predictions
