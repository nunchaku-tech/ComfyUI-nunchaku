Radial Attention
================

The Radial Attention feature provides optimized sparse attention patterns for video generation models, significantly reducing computations and latencies while maintaining generation quality.

Overview
--------

Radial Attention implements training-free sparse attention acceleration specifically designed for WAN video generation models. It uses a static sparsity pattern with spatial and temporal decay as tokens become distant, that preserve local and temporal relationships while reducing attention computation and complexity for video sequences.

**Key Benefits:**

* **Speed Optimized**: Accelerates inference through sparse computation patterns
* **Quality Preserved**: Maintains visual quality through spatiotemporal-decay-aware sparsity pattern
* **Training-Free**: No model retraining required - works with existing WAN models
* **Sparsity+Quantization**: Combines sparse attention with quantization for further performance gains

Features
--------

**Timestep-Aware Switching**
  Use dense attention for early denoising steps (high-frequency details) and sparse attention for later steps (refinement).

**Configurable Sparsity Patterns**
  Adjust decay factors to control the sparsity/quality tradeoff.

**Compatibility with SageAttention (Quantized Attention)**
  Leverages SageAttention v1/v2 backends for more efficient low-bit sparse attention computation.

**Automatic Fallbacks**
  Seamlessly falls back to dense attention for cross-attention layers and incompatible operations.

Nodes
-----

PatchRadialAttention
~~~~~~~~~~~~~~~~~~~~

Applies radial attention optimization to WAN diffusion models.

**Parameters:**

* **model** (MODEL): Input WAN diffusion model
* **dense_layers** (STRING): Comma-separated layer indices to apply radial attention (e.g., "0,1,2")
* **dense_timesteps** (INT): Apply radial attention from this timestep onwards (0-100)
* **block_size** (64|128): Attention block size - larger blocks are faster but less precise
* **decay_factor** (FLOAT): Attention decay factor (0.0-2.0) - lower is faster, higher is more accurate

**Returns:**
* **model** (MODEL): Model with radial attention patches applied

**Usage Example:**

.. code-block:: json

   {
     "inputs": {
       "model": ["Load WAN Model", 0],
       "dense_layers": "0",
       "dense_timesteps": 7,
       "block_size": 128,
       "decay_factor": 0.2
     }
   }

Installation Requirements
--------------------------

Radial Attention requires additional sparse attention backends:

**Option 1: Block-Sparse-SageAttention-2.0 (Recommended)**

.. code-block:: bash
   git clone https://github.com/thu-ml/SpargeAttn nodes/radial_attn/third_party/Block-Sparse-SageAttention-2.0
   # Navigate to the third-party directory
   cd nodes/radial_attn/third_party/Block-Sparse-SageAttention-2.0

   # Install dependencies
   pip install ninja torch torchvision transformers diffusers einops

   # Install the package
   python setup.py install

**Option 2: Sparse_SageAttention_API**

.. code-block:: bash
   git clone https://github.com/jt-zhang/Sparse_SageAttention_API nodes/radial_attn/third_party/Sparse_SageAttention_API
   # Navigate to the API directory
   cd nodes/radial_attn/third_party/Sparse_SageAttention_API

   # Install the package
   python setup.py install

**System Requirements:**

* Python >= 3.9
* PyTorch >= 2.3.0
* CUDA >= 12.0 (12.4+ recommended for fp8 support)
* Compatible GPU: RTX 30/40/50 series, A100, H100

Configuration Guide
--------------------

**Layer Selection Strategy:**

* **Early Dense Layers (0)**: Based on extensive parameter search, we found that keeping dense attention in the first layer (layer 0) provides the best balance of speed and quality.

**Timestep Configuration:**

* **dense_timesteps=num_inference_steps // 4**:
  This setting applies radial attention starting from the quarter of the total inference steps, allowing for high-frequency details to be precisely captured in the initial denoising steps while optimizing later steps.

Performance Optimization
-------------------------

**Speed Improvements:**

* up to 10x faster attention computation
* up to 2.5x end-to-end inference speedup
* Greater speedup for longer sequences
* Automatic fallback prevents slowdowns

**Quality Preservation:**

* Minimal quality loss with properly tuned parameters
* Maintains spatial/temporal consistency in video generation

Technical Details
-----------------

**Attention Pattern:**

Radial attention uses a spatiotemporal-decay-aware sparsity pattern that:

* Maintains full attention within identical/adjacent frames
* Applies spatial/temporal decay to distant frames (smaller attention band width)
* Preserves spatial and temporal relationships that are necessary for coherent video generation

**Implementation:**

* Built on SageAttention v1/v2 backends
* CUDA kernel optimizations for RTX and data center GPUs
* Block-sparse computation with hardware friendliness

**Compatibility:**

* Works with WAN 2.1 i2v models
* Supports multiple batch sizes and resolutions
* Compatible with LoRA and ControlNet extensions
* Maintains ComfyUI workflow compatibility
