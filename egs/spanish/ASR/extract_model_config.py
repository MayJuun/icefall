import torch
import os
import re
from collections import defaultdict
import json

# Expand the home directory path
checkpoint_path = os.path.expanduser("/home/grey_faulkenberry_mayjuun_com/asr-projects/models/icefall-asr-commonvoice-fr-pruned-transducer-stateless7-streaming-2023-04-02/exp/pretrained.pt")
ckpt = torch.load(checkpoint_path, map_location='cpu')

# Initialize containers for different parameters
config = {
    "encoder_dims": set(),
    "feedforward_dims": set(),
    "kernel_sizes": set(),
    "attention_dims": set(),
    "num_heads": set(),
    "layer_structure": defaultdict(dict)
}

# Track tensor shapes for debugging
tensor_shapes = {}

# Process and extract configuration
print("Analyzing model weights and structure...")
for key, tensor in ckpt["model"].items():
    if not isinstance(tensor, torch.Tensor):
        continue
        
    # Store tensor shape for debugging
    tensor_shapes[key] = list(tensor.shape)
    
    # Extract encoder dimensions
    if "encoder.encoders" in key and "weight" in key:
        shape = tensor.shape
        if len(shape) == 2:
            for dim in shape:
                if dim > 100 and dim < 1000:  # Filter likely encoder dimensions
                    config["encoder_dims"].add(dim)
    
    # Extract feedforward dimensions
    if "feed_forward" in key and "linear" in key and "weight" in key:
        shape = tensor.shape
        for dim in shape:
            if dim > 500:  # Typical feedforward dimensions are larger
                config["feedforward_dims"].add(dim)
    
    # Extract kernel sizes from CNN modules
    if "cnn_module" in key and "conv" in key and "weight" in key:
        if len(tensor.shape) >= 3:  # Ensure it's a conv layer
            kernel_size = tensor.shape[2]  # Conv kernel size is in the 3rd dimension
            if kernel_size > 1:
                config["kernel_sizes"].add(kernel_size)
    
    # Extract layer-specific information
    match = re.search(r'encoder\.encoders\.(\d+)\.', key)
    if match:
        layer_idx = int(match.group(1))
        if "feed_forward" in key and "linear1.weight" in key:
            config["layer_structure"][layer_idx]["ff_in"] = tensor.shape[1]
            config["layer_structure"][layer_idx]["ff_out"] = tensor.shape[0]
        elif "feed_forward" in key and "linear2.weight" in key:
            config["layer_structure"][layer_idx]["ff_hidden"] = tensor.shape[1]
        elif "self_attn" in key and "q_proj.weight" in key:
            config["layer_structure"][layer_idx]["attn_dim"] = tensor.shape[1]
            config["attention_dims"].add(tensor.shape[1])
        elif "self_attn" in key and "out_proj.weight" in key:
            config["layer_structure"][layer_idx]["num_heads"] = tensor.shape[1] // tensor.shape[0]
        elif "cnn_module" in key and "conv" in key and "weight" in key and len(tensor.shape) >= 3:
            config["layer_structure"][layer_idx]["kernel_size"] = tensor.shape[2]

# Find keys with potentially useful model configuration information
print("\nSearching for configuration parameters...")
if "model_args" in ckpt:
    print("Found model_args in checkpoint")
    for key, value in ckpt["model_args"].items():
        print(f"  {key}: {value}")

if "model_config" in ckpt:
    print("Found model_config in checkpoint")
    for key, value in ckpt["model_config"].items():
        print(f"  {key}: {value}")

# Search for all interesting layers and structures
encoder_layers = {k: v for k, v in tensor_shapes.items() if "encoder" in k}
decoder_layers = {k: v for k, v in tensor_shapes.items() if "decoder" in k}
joiner_layers = {k: v for k, v in tensor_shapes.items() if "joiner" in k}

# Format the results
print("\n==== Zipformer Model Configuration ====")
print(f"Encoder dimensions: {sorted(config['encoder_dims'])}")
print(f"Feedforward dimensions: {sorted(config['feedforward_dims'])}")
print(f"Kernel sizes: {sorted(config['kernel_sizes'])}")
print(f"Attention dimensions: {sorted(config['attention_dims'])}")
print(f"Number of attention heads: {sorted(config['num_heads'])}")

print("\n==== Layer-by-Layer Structure ====")
for layer, params in sorted(config["layer_structure"].items()):
    print(f"Layer {layer}:")
    for param_name, value in params.items():
        print(f"  {param_name}: {value}")

# Generate configuration strings for baseline.sh
encoder_dims = sorted([d for d in config["encoder_dims"] if 300 < d < 500])
if encoder_dims:
    encoder_dim_str = ",".join([str(d) for d in encoder_dims] * 6)  # Assuming 6 layers
    print(f"\n==== Suggested Configuration Parameters ====")
    print(f"encoder-dim: \"{encoder_dim_str}\"")
    print(f"encoder-unmasked-dim: \"{encoder_dim_str}\"")

kernel_sizes = sorted(config["kernel_sizes"])
if kernel_sizes:
    kernel_str = ",".join([str(k) for k in kernel_sizes] * 6)  # Assuming 6 layers
    print(f"cnn-module-kernel: \"{kernel_str}\"")

ff_dims = sorted(config["feedforward_dims"])
if ff_dims and len(ff_dims) >= 2:
    ff_dim_str = ",".join([str(d) for d in ff_dims])
    print(f"feedforward-dim: \"{ff_dim_str}\"")

# Dump detailed tensor structures
print("\n==== Key Model Component Shapes ====")

# Find modules with specific structures
print("Encoder layers:")
encoder_pattern = re.compile(r'encoder\.encoders\.(\d+)\.')
for key, shape in sorted(tensor_shapes.items()):
    if encoder_pattern.search(key) and ('linear' in key or 'conv' in key) and 'weight' in key:
        print(f"  {key}: {shape}")

print("\nFeedforward components:")
for key, shape in sorted(tensor_shapes.items()):
    if 'feed_forward' in key and 'weight' in key:
        print(f"  {key}: {shape}")

print("\nCNN components:")
for key, shape in sorted(tensor_shapes.items()):
    if 'cnn_module' in key and 'weight' in key:
        print(f"  {key}: {shape}")

# Save the full configuration as JSON for reference
with open("model_config.json", "w") as f:
    # Convert sets to lists for JSON serialization
    serializable_config = {k: list(v) if isinstance(v, set) else v for k, v in config.items()}
    # Convert defaultdict to dict
    serializable_config["layer_structure"] = dict(serializable_config["layer_structure"])
    # Add tensor shapes for reference
    serializable_config["tensor_shapes"] = tensor_shapes
    json.dump(serializable_config, f, indent=2)
print("\nFull configuration saved to model_config.json")
