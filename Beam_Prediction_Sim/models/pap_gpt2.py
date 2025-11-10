"""
Prompt-as-Prefix (PaP) and GPT-2 integration
Improved with learnable text prototypes and better prompting
"""
import torch
import torch.nn as nn
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def build_pap_prompts(U: int, H: int) -> List[str]:
    """Build domain knowledge, instruction, and statistics prompts
    
    Args:
        U: past sequence length
        H: future prediction horizon
    
    Returns:
        prompts: list of prompt strings
    """
    domain = (
        "Task: Predict future optimal beam directions (angles of departure) "
        "for mmWave communication from past observations of user movement "
        "and beam indices."
    )
    
    instruction = (
        f"Given the past U={U} time steps of beam indices and angles, "
        f"predict the next H={H} time steps of optimal beam directions."
    )
    
    # Statistics will be filled at runtime
    statistics = "Statistics will be provided at runtime."
    
    return [domain, instruction, statistics]


def load_gpt2_model(
    model_name: str = "gpt2",
    freeze: bool = True
) -> Tuple[Optional[object], Optional[object]]:
    """Load GPT-2 tokenizer and model
    
    Args:
        model_name: GPT-2 model variant
        freeze: whether to freeze GPT-2 weights
    
    Returns:
        tokenizer, model (or None, None if failed)
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        
        logger.info(f"Loaded GPT-2 model: {model_name}, freeze={freeze}")
        return tokenizer, model
    
    except Exception as e:
        logger.warning(f"Failed to load GPT-2: {e}")
        return None, None


def embed_text_with_gpt2(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device
) -> torch.Tensor:
    """Embed texts using GPT-2
    
    Args:
        texts: list of text strings
        tokenizer: GPT-2 tokenizer
        model: GPT-2 model
        device: torch device
    
    Returns:
        embeddings: [len(texts), D] where D is GPT-2 hidden dim
    """
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        embeddings = []
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # Get hidden states
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # [1, seq_len, D]
            
            # Mean pooling over sequence
            embed = hidden.mean(dim=1)  # [1, D]
            embeddings.append(embed)
        
        embeddings = torch.cat(embeddings, dim=0)  # [len(texts), D]
    
    return embeddings


class LearnableTextPrototypes(nn.Module):
    """Learnable text prototypes for patch reprogramming
    
    Instead of using the full GPT-2 vocabulary, we learn a small set
    of prototype embeddings that best represent time series patterns.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Args:
            vocab_size: number of learnable prototypes
            embedding_dim: dimension (should match GPT-2 hidden dim)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Learnable prototypes initialized with small random values
        self.prototypes = nn.Parameter(
            torch.randn(vocab_size, embedding_dim) / (embedding_dim ** 0.5)
        )
    
    def forward(self) -> torch.Tensor:
        """Return prototypes
        
        Returns:
            prototypes: [vocab_size, embedding_dim]
        """
        return self.prototypes
    
    def get_nearest_prototype_indices(
        self,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Find nearest prototype for each embedding (optional)
        
        Args:
            embeddings: [B, ..., D] embeddings
        
        Returns:
            indices: [B, ...] nearest prototype indices
        """
        # Flatten batch dimensions
        orig_shape = embeddings.shape[:-1]
        embeddings_flat = embeddings.view(-1, self.embedding_dim)  # [N, D]
        
        # Compute distances to prototypes
        # ||e - p||^2 = ||e||^2 + ||p||^2 - 2 <e, p>
        e_norm = (embeddings_flat ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        p_norm = (self.prototypes ** 2).sum(dim=1, keepdim=True).t()  # [1, V]
        dot = embeddings_flat @ self.prototypes.t()  # [N, V]
        
        distances = e_norm + p_norm - 2 * dot  # [N, V]
        
        # Find nearest
        indices = distances.argmin(dim=1)  # [N]
        
        # Reshape
        indices = indices.view(*orig_shape)
        
        return indices


class PatchReprogramming(nn.Module):
    """Patch reprogramming using cross-attention with text prototypes
    
    This module converts time series patch embeddings into text-like
    representations that align with GPT-2's input space.
    """
    
    def __init__(
        self,
        d_model: int,
        gpt2_dim: int,
        vocab_size: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: dimension of patch embeddings
            gpt2_dim: dimension of GPT-2 embeddings
            vocab_size: number of learnable text prototypes
            n_heads: number of attention heads
            dropout: dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.gpt2_dim = gpt2_dim
        self.vocab_size = vocab_size
        
        # Learnable text prototypes
        self.text_prototypes = LearnableTextPrototypes(vocab_size, gpt2_dim)
        
        # Multi-head cross-attention
        # Query: from patch embeddings
        # Key, Value: from text prototypes
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            kdim=gpt2_dim,
            vdim=gpt2_dim,
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Projection to GPT-2 dimension
        self.proj_to_gpt2 = nn.Linear(d_model, gpt2_dim)
    
    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Reprogram patches to text-like representations
        
        Args:
            patch_embeddings: [B, P, d_model] patch embeddings
        
        Returns:
            reprogrammed: [B, P, gpt2_dim] text-like representations
        """
        B, P, _ = patch_embeddings.shape
        
        # Get text prototypes [V, gpt2_dim]
        prototypes = self.text_prototypes()
        
        # Expand prototypes for batch
        # Key, Value: [B, V, gpt2_dim]
        key = prototypes.unsqueeze(0).expand(B, -1, -1)
        value = key
        
        # Query: patch embeddings [B, P, d_model]
        query = patch_embeddings
        
        # Cross-attention
        attn_out, _ = self.cross_attn(query, key, value)  # [B, P, d_model]
        
        # Residual and norm
        out = self.norm(patch_embeddings + attn_out)  # [B, P, d_model]
        
        # Project to GPT-2 dimension
        reprogrammed = self.proj_to_gpt2(out)  # [B, P, gpt2_dim]
        
        return reprogrammed


class PromptAsPrefix(nn.Module):
    """Prompt-as-Prefix: concatenate prompt embeddings with patch embeddings
    
    This enriches the input with domain knowledge and statistics.
    """
    
    def __init__(self, gpt2_dim: int):
        """
        Args:
            gpt2_dim: GPT-2 hidden dimension
        """
        super().__init__()
        self.gpt2_dim = gpt2_dim
        
        # We'll store prompt embeddings (computed once)
        self.register_buffer("prompt_embeddings", torch.zeros(0, gpt2_dim))
    
    def set_prompt_embeddings(self, embeddings: torch.Tensor):
        """Set prompt embeddings (from GPT-2)
        
        Args:
            embeddings: [num_prompts, gpt2_dim]
        """
        self.prompt_embeddings = embeddings
    
    def forward(
        self,
        patch_embeddings: torch.Tensor,
        stats_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Concatenate prompts with patches
        
        Args:
            patch_embeddings: [B, P, gpt2_dim]
            stats_embedding: [B, 1, gpt2_dim] optional statistics embedding
        
        Returns:
            combined: [B, num_prompts + P (+ 1), gpt2_dim]
        """
        B, P, D = patch_embeddings.shape
        
        # Expand prompts for batch
        prompts = self.prompt_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        
        if stats_embedding is not None:
            # Concatenate: [prompts, stats, patches]
            combined = torch.cat([prompts, stats_embedding, patch_embeddings], dim=1)
        else:
            # Concatenate: [prompts, patches]
            combined = torch.cat([prompts, patch_embeddings], dim=1)
        
        return combined
    
    def get_num_prompts(self) -> int:
        """Get number of prompt tokens"""
        return self.prompt_embeddings.size(0)


class GPT2Backbone(nn.Module):
    """GPT-2 backbone for processing reprogrammed embeddings
    
    This is a frozen GPT-2 model that processes the prompt + patch embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        freeze: bool = True
    ):
        """
        Args:
            model_name: GPT-2 model variant
            freeze: whether to freeze weights
        """
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        
        # Load GPT-2
        tokenizer, model = load_gpt2_model(model_name, freeze)
        
        if model is None:
            raise RuntimeError(f"Failed to load GPT-2 model: {model_name}")
        
        self.tokenizer = tokenizer
        self.gpt2 = model
        self.hidden_dim = model.config.n_embd
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Process embeddings through GPT-2
        
        Args:
            embeddings: [B, seq_len, hidden_dim] input embeddings
        
        Returns:
            outputs: [B, seq_len, hidden_dim] output hidden states
        """
        # Move model to same device as input
        device = embeddings.device
        self.gpt2.to(device)
        
        if self.freeze:
            self.gpt2.eval()
            with torch.no_grad():
                outputs = self.gpt2(inputs_embeds=embeddings)
                hidden_states = outputs.last_hidden_state
        else:
            outputs = self.gpt2(inputs_embeds=embeddings)
            hidden_states = outputs.last_hidden_state
        
        return hidden_states
    
    def embed_text(self, text: str, device: torch.device) -> torch.Tensor:
        """Embed a text string
        
        Args:
            text: input text
            device: torch device
        
        Returns:
            embedding: [1, hidden_dim] text embedding (mean pooled)
        """
        return embed_text_with_gpt2([text], self.tokenizer, self.gpt2, device)
