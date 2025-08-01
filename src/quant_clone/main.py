from gguf import GGUFReader, GGMLQuantizationType, ReaderTensor
import sys

IGNORE = [
    "_norm.weight",
    "ffn_gate_inp.weight",
    "altup",
    "laurel",
    "per_layer_model_proj",
    "ssm_conv1d.weight",
    "shortconv.conv.weight",
    "time_mix_first.weight",
    "time_mix_w0.weight",
    "time_mix_w1.weight",
    "time_mix_w2.weight",
    "time_mix_v0.weight",
    "time_mix_v1.weight",
    "time_mix_v2.weight",
    "time_mix_a0.weight",
    "time_mix_a1.weight",
    "time_mix_a2.weight",
    "time_mix_g1.weight",
    "time_mix_g2.weight",
    "time_mix_decay_w1.weight",
    "time_mix_decay_w2.weight",
    "time_mix_lerp_fused.weight",
    "attn_rel_b.weight",
]

# adapted from https://github.com/unslothai/llama.cpp/blob/master/src/llama-quant.cpp#L818
def ignore_tensor(tensor: ReaderTensor):
    if not tensor.name.endswith("weight"):
        return True
    # quantize &= (ggml_n_dims(tensor) >= 2);
    if len(tensor.shape) >= 3:  # is this right?
        return True
    if any(s in tensor.name for s in IGNORE):
        return True
    return False

def run():
    args = sys.argv[1:]
    if not args:
        print("Usage: quant_clone <GGUF file> <output (default=cmd.txt)>")
        print("Error: No filename provided.")
        sys.exit(1)
    else:
        file = "cmd.txt"
        if len(args) > 1:
            file = args[1]
        try:
            reader = GGUFReader(args[0], "r")
            cmd = "llama-quantize --imatrix <imatrix_unsloth.dat>"
            for tensor in reader.tensors:
                if ignore_tensor(tensor):
                    continue
                cmd = f"{cmd} --tensor-type {tensor.name}={GGMLQuantizationType(tensor.tensor_type).name}"
            cmd = f"{cmd} <input.gguf> <output.gguf> Q8_0"
            print(f"{cmd}")
            with open(file, "w") as f:
                f.write(cmd)
            print(f"Command saved to {file}")
        except Exception as e:
            print(f"Exception: {e}")
            sys.exit(1)