"""
Radial Attention Core Implementation

This module provides optimized radial sparse attention for video generation models.
It implements efficient attention patterns that reduce computational complexity
while maintaining generation quality for long video sequences.

Key Features:
    - Radial sparse attention patterns
    - Block-based attention computation
    - CUDA architecture-aware optimizations
    - Fallback to dense attention when needed
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, List


def get_cuda_arch_versions() -> List[str]:
    """
    Get CUDA architecture versions for all available devices.
    
    Returns:
        List[str]: List of CUDA architecture strings (e.g., ['sm86', 'sm90'])
    """
    cuda_archs = []
    for device_idx in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(device_idx)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs


# Try to import sparse attention backends
has_spas_sage_attn = False
try:
    from spas_sage_attn import block_sparse_sage2_attn_cuda
    SPARSE_SAGE_AVAILABLE = True
    has_spas_sage_attn = True
except ImportError:
    try:
        from sparse_sageattn import sparse_sageattn as block_sparse_sage2_attn_cuda
        SPARSE_SAGE_AVAILABLE = True
    except ImportError:
        SPARSE_SAGE_AVAILABLE = False
        print("Sparse SageAttention not available, using dense attention fallback")


def sparge_mask_convert(mask: torch.Tensor, block_size: int = 128, arch="sm") -> torch.Tensor:
    assert block_size in [128, 64], "Radial Attention only supports block size of 128 or 64"
    assert mask.shape[0] == mask.shape[1], "Input mask must be square."

    if block_size == 128:
        if arch == "sm90" and has_spas_sage_attn:
            new_mask = torch.repeat_interleave(mask, 2, dim=0)
        else:
            new_mask = torch.repeat_interleave(mask, 2, dim=1)
        
    elif block_size == 64:
        if arch == "sm90" and has_spas_sage_attn:
            num_row, num_col = mask.shape
            reshaped_mask = mask.view(num_row, num_col // 2, 2)
            new_mask = torch.max(reshaped_mask, dim=2).values
        else:
            num_row, num_col = mask.shape
            reshaped_mask = mask.view(num_row // 2, 2, num_col)
            new_mask = torch.max(reshaped_mask, dim=1).values

    return new_mask


def shrink_mask_strict(mask: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Shrink attention mask to block-level granularity with strict density filtering.
    
    This function converts token-level attention masks to block-level masks by
    analyzing the density of connections within each block and applying
    strict filtering criteria.
    
    Args:
        mask (torch.Tensor): Token-level attention mask
        block_size (int): Size of attention blocks
        
    Returns:
        torch.Tensor: Block-level attention mask
    """
    seq_len = mask.shape[0]
    block_num = seq_len // block_size
    
    # Reshape mask to block structure
    mask_blocks = mask[:block_num * block_size, :block_num * block_size]
    mask_blocks = mask_blocks.view(block_num, block_size, block_num, block_size)
    
    # Calculate column densities within each block
    col_densities = mask_blocks.sum(dim=1) / block_size
    
    # Identify high-density columns and compute fraction
    non_zero_densities = col_densities > 0
    high_density_cols = col_densities > 1/3
    frac_high_density_cols = high_density_cols.sum(dim=-1) / (non_zero_densities.sum(dim=-1) + 1e-9)
    
    # Create block mask based on density threshold
    block_mask = frac_high_density_cols > 0.6
    
    # Always include first and last blocks for stability
    if block_mask.numel() > 0:
        block_mask[0] = True
        block_mask[-1] = True
    
    return block_mask


def get_diagonal_split_mask(i: int, j: int, token_per_frame: int, sparse_type: str, query: torch.Tensor) -> torch.Tensor:
    """
    Generate diagonal split mask for frame-to-frame attention.
    
    This function creates attention masks that control connections between different
    video frames based on their temporal distance and sparse attention patterns.
    
    Args:
        i (int): Source frame index
        j (int): Target frame index  
        token_per_frame (int): Number of tokens per frame
        sparse_type (str): Type of sparse attention pattern ("radial")
        query (torch.Tensor): Query tensor for device placement
        
    Returns:
        torch.Tensor: Boolean mask for frame-to-frame attention [token_per_frame, token_per_frame]
    """
    assert sparse_type in ["radial"], f"Unsupported sparse type: {sparse_type}"
    
    dist = abs(i - j)
    group = dist.bit_length()
    threshold = 128  # Hardcoded threshold equal to block size
    
    decay_length = 2 ** token_per_frame.bit_length() / 2 ** group
    
    if decay_length >= threshold:
        return torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
    
    split_factor = int(threshold / decay_length)
    modular = dist % split_factor
    
    if modular == 0:
        return torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
    else:
        return torch.zeros((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)


def get_window_width(i: int, j: int, token_per_frame: int, sparse_type: str, num_frame: int, 
                    decay_factor: float = 1.0, block_size: int = 128, model_type: str = None) -> float:
    """
    Calculate attention window width for frame pair (i, j).
    
    This function determines the spatial attention window size based on temporal
    distance between frames, model type, and decay parameters.
    
    Args:
        i (int): Source frame index
        j (int): Target frame index
        token_per_frame (int): Number of tokens per frame
        sparse_type (str): Sparse attention pattern type
        num_frame (int): Total number of frames
        decay_factor (float): Controls attention decay over distance
        block_size (int): Attention block size
        model_type (str): Model type ("wan" or "hunyuan")
        
    Returns:
        float: Attention window width
        
    Raises:
        ValueError: If model_type is not supported
    """
    assert sparse_type in ["radial"], f"Unsupported sparse type: {sparse_type}"
    
    dist = abs(i - j)
    
    # Model-specific attention patterns
    if model_type == "wan":
        if dist < 1:
            return token_per_frame
        if dist == 1:
            return token_per_frame // 2
    elif model_type == "hunyuan":
        if dist <= 1:
            return token_per_frame
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate decay-based window width
    group = dist.bit_length()
    decay_length = 2 ** token_per_frame.bit_length() / 2 ** group * decay_factor
    threshold = block_size
    
    return max(decay_length, threshold)


def pad_qkv(input_tensor: torch.Tensor, padding_gran: int = 128) -> torch.Tensor:
    """
    Pad input tensor to nearest multiple of padding granularity.
    
    This ensures tensor dimensions are compatible with block-based attention
    computation requirements.
    
    Args:
        input_tensor (torch.Tensor): Input tensor [batch_size, num_heads, seq_len, hidden_dim]
        padding_gran (int): Padding granularity
        
    Returns:
        torch.Tensor: Padded tensor with seq_len padded to multiple of padding_gran
    """
    batch_size, num_heads, seq_len, hidden_dim = input_tensor.shape
    padded_seq_len = ((seq_len + padding_gran - 1) // padding_gran) * padding_gran
    
    if padded_seq_len == seq_len:
        return input_tensor
    
    padding_size = padded_seq_len - seq_len
    return F.pad(input_tensor, (0, 0, 0, padding_size), mode='constant', value=0)


def generate_log_mask_shrinked(query: torch.Tensor, seq_len: int, video_token_num: int, 
                              num_frame: int, block_size: int = 128, sparse_type: str = "radial", 
                              decay_factor: float = 0.5, model_type: str = None) -> torch.Tensor:
    """
    Generate shrinked attention mask for video sequences.
    
    This function creates block-level attention masks by processing frame pairs
    individually to manage memory usage efficiently. It applies radial attention
    patterns with configurable decay and model-specific optimizations.
    
    Args:
        query (torch.Tensor): Query tensor for device placement
        seq_len (int): Total sequence length
        video_token_num (int): Number of video tokens
        num_frame (int): Number of video frames
        block_size (int): Attention block size
        sparse_type (str): Sparse attention pattern type
        decay_factor (float): Attention decay factor
        model_type (str): Model type for specific optimizations
        
    Returns:
        torch.Tensor: Block-level attention mask [seq_len//block_size, seq_len//block_size]
    """
    final_log_mask = torch.zeros((seq_len // block_size, seq_len // block_size), 
                                device=query.device, dtype=torch.bool)
    token_per_frame = video_token_num // num_frame
    video_text_border = video_token_num // block_size

    # Prepare spatial indices for window-based masking
    col_indices = torch.arange(0, token_per_frame, device=query.device).view(1, -1)
    row_indices = torch.arange(0, token_per_frame, device=query.device).view(-1, 1)
    
    # Enable full attention for text tokens
    final_log_mask[video_text_border:] = True
    final_log_mask[:, video_text_border:] = True
    
    # Process each frame pair
    for i in range(num_frame):
        for j in range(num_frame):
            local_mask = torch.zeros((token_per_frame, token_per_frame), 
                                   device=query.device, dtype=torch.bool)
            
            # Special handling for attention sink (first frame in WAN)
            if j == 0 and model_type == "wan":
                local_mask = torch.ones((token_per_frame, token_per_frame), 
                                      device=query.device, dtype=torch.bool)
            else:
                # Apply window-based attention pattern
                window_width = get_window_width(i, j, token_per_frame, sparse_type, num_frame, 
                                              decay_factor=decay_factor, block_size=block_size, 
                                              model_type=model_type)
                local_mask = torch.abs(col_indices - row_indices) <= window_width
                
                # Apply diagonal split mask for additional sparsity
                split_mask = get_diagonal_split_mask(i, j, token_per_frame, sparse_type, query)
                local_mask = torch.logical_and(local_mask, split_mask)

            # Calculate padding and block positions
            remainder_row = (i * token_per_frame) % block_size
            remainder_col = (j * token_per_frame) % block_size
            
            all_length_row = remainder_row + ((token_per_frame - 1) // block_size + 1) * block_size
            all_length_col = remainder_col + ((token_per_frame - 1) // block_size + 1) * block_size
            
            # Create padded local mask
            padded_local_mask = torch.zeros((all_length_row, all_length_col), 
                                          device=query.device, dtype=torch.bool)
            padded_local_mask[remainder_row:remainder_row + token_per_frame, 
                            remainder_col:remainder_col + token_per_frame] = local_mask
            
            # Shrink to block level and merge with final mask
            block_mask = shrink_mask_strict(padded_local_mask, block_size=block_size)
            
            block_row_start = (i * token_per_frame) // block_size
            block_col_start = (j * token_per_frame) // block_size
            block_row_end = block_row_start + block_mask.shape[0]
            block_col_end = block_col_start + block_mask.shape[1]
            
            final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end] = torch.logical_or(
                final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end], block_mask)
    
    # Print sparsity information
    sparsity = 1 - final_log_mask.sum().item() / final_log_mask.numel()
    print(f"Attention mask sparsity: {sparsity:.3f}")
    
    return final_log_mask


class MaskMap:
    """
    Caching manager for attention masks.
    
    This class provides efficient caching and retrieval of attention masks
    to avoid recomputation during inference. It maintains a single cached
    mask that can be reused across attention layers.
    
    Attributes:
        _log_mask (torch.Tensor, optional): Cached attention mask
        video_token_num (int): Number of video tokens
        num_frame (int): Number of video frames
    """
    
    _log_mask: Optional[torch.Tensor] = None

    def __init__(self, video_token_num: int = 25440, num_frame: int = 16):
        """
        Initialize mask map with video parameters.
        
        Args:
            video_token_num (int): Total number of video tokens
            num_frame (int): Number of video frames
        """
        self.video_token_num = video_token_num
        self.num_frame = num_frame

    def query_log_mask(self, query: torch.Tensor, sparse_type: str, block_size: int = 128, 
                      decay_factor: float = 0.5, model_type: str = None) -> torch.Tensor:
        """
        Query or generate attention mask.
        
        This method returns a cached mask if available, or generates a new one
        using the provided parameters.
        
        Args:
            query (torch.Tensor): Query tensor for device and shape information
            sparse_type (str): Type of sparse attention pattern
            block_size (int): Attention block size
            decay_factor (float): Attention decay factor
            model_type (str): Model type for optimization
            
        Returns:
            torch.Tensor: Block-level attention mask
        """
        if MaskMap._log_mask is None:
            # Determine sequence length from query tensor
            seq_len = query.shape[-2] if query.dim() == 4 else query.shape[0]
            
            # Generate new mask
            MaskMap._log_mask = generate_log_mask_shrinked(
                query, seq_len, self.video_token_num, self.num_frame, 
                sparse_type=sparse_type, decay_factor=decay_factor, 
                model_type=model_type, block_size=block_size
            )
        
        return MaskMap._log_mask


def dense_attention_fallback(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    Fallback to dense attention computation.
    
    This function provides a reliable fallback when sparse attention backends
    are unavailable or fail. It uses PyTorch's optimized scaled dot-product attention.
    
    Args:
        query (torch.Tensor): Query tensor [batch_size, num_heads, seq_len, hidden_dim]
        key (torch.Tensor): Key tensor [batch_size, num_heads, seq_len, hidden_dim]
        value (torch.Tensor): Value tensor [batch_size, num_heads, seq_len, hidden_dim]
        
    Returns:
        torch.Tensor: Attention output with same shape as query
    """
    return F.scaled_dot_product_attention(query, key, value, is_causal=False)


def sparse_sage_attention_backend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                 mask_map: Optional[MaskMap] = None, video_mask: Optional[torch.Tensor] = None, 
                                 block_size: int = 128) -> torch.Tensor:
    """
    Sparse attention backend using SageAttention.
    
    This function applies sparse attention using the SageAttention library
    with automatic fallback to dense attention if unavailable or on error.
    
    Args:
        query (torch.Tensor): Query tensor [batch_size, num_heads, seq_len, hidden_dim]
        key (torch.Tensor): Key tensor [batch_size, num_heads, seq_len, hidden_dim]  
        value (torch.Tensor): Value tensor [batch_size, num_heads, seq_len, hidden_dim]
        mask_map (MaskMap, optional): Mask map instance (unused in current implementation)
        video_mask (torch.Tensor, optional): Block-level attention mask
        block_size (int): Attention block size
        
    Returns:
        torch.Tensor: Attention output with same shape as query
    """
    batch_size, num_heads, seq_len, hidden_dim = query.shape
    
    if not SPARSE_SAGE_AVAILABLE:
        print("Sparse attention backend not available, falling back to dense attention")
        return dense_attention_fallback(query, key, value)
    
    try:
        # Get CUDA architecture for mask conversion
        arch = get_cuda_arch_versions()[0] if get_cuda_arch_versions() else "sm80"
        
        # Convert mask format for sparse attention
        converted_mask = sparge_mask_convert(mask=video_mask, block_size=block_size, arch=arch)
        converted_mask = repeat(converted_mask, "s t -> b h s t", b=batch_size, h=num_heads)
        converted_mask = converted_mask.to(torch.int8)
        
        # Apply sparse attention
        output = block_sparse_sage2_attn_cuda(
            query, key, value,
            mask_id=converted_mask.contiguous(),
            tensor_layout="HND",
        )
        
        return output
        
    except Exception as e:
        print(f"Sparse attention computation failed: {e}")
        print("Falling back to dense attention")
        return dense_attention_fallback(query, key, value)


def RadialAttention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   mask_map: Optional[MaskMap] = None, sparsity_type: str = "radial", 
                   block_size: int = 128, decay_factor: float = 1.0, 
                   model_type: str = None) -> torch.Tensor:
    """
    Main radial attention function.
    
    This is the primary interface for radial sparse attention computation.
    It handles padding, mask generation, and delegates to the appropriate
    attention backend.
    
    Args:
        query (torch.Tensor): Query tensor [batch_size, num_heads, seq_len, hidden_dim]
        key (torch.Tensor): Key tensor [batch_size, num_heads, seq_len, hidden_dim]
        value (torch.Tensor): Value tensor [batch_size, num_heads, seq_len, hidden_dim]
        mask_map (MaskMap, optional): Mask map for attention mask caching
        sparsity_type (str): Type of sparse attention pattern
        block_size (int): Attention block size
        decay_factor (float): Attention decay factor
        model_type (str): Model type for optimization
        
    Returns:
        torch.Tensor: Attention output [batch_size, num_heads, seq_len, hidden_dim]
    """
    batch_size, num_heads, seq_len, hidden_dim = query.shape
    
    # Pad tensors to block size granularity
    query_padded = pad_qkv(query, padding_gran=block_size)
    key_padded = pad_qkv(key, padding_gran=block_size)
    value_padded = pad_qkv(value, padding_gran=block_size)
    new_seq_len = query_padded.shape[-2]

    # Generate or retrieve attention mask
    if sparsity_type == "dense":
        video_mask = torch.ones((new_seq_len // block_size, new_seq_len // block_size), 
                               device=query.device, dtype=torch.bool)
    else:
        video_mask = (mask_map.query_log_mask(query_padded, sparsity_type, 
                                            block_size=block_size, decay_factor=decay_factor, 
                                            model_type=model_type) if mask_map else None)
    
    # Apply sparse attention and return original sequence length
    output_padded = sparse_sage_attention_backend(
        query_padded, key_padded, value_padded, 
        mask_map=mask_map, video_mask=video_mask, block_size=block_size
    )
    
    return output_padded[:, :, :seq_len, :]