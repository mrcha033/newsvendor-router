"""Inspect SDPA dispatch for representative ModernBERT masks without attention execution."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from torch.backends.cuda import SDPAParams
    from torch.nn.attention import SDPBackend
    from transformers import AutoConfig
    from transformers.masking_utils import (
        create_bidirectional_mask,
        create_bidirectional_sliding_window_mask,
    )

    from newsvendor.io import digest, read, require, write

    require(not Path(args.output).exists(), "Preserve earlier diagnostics")
    require(torch.cuda.device_count() == 1 and torch.cuda.get_device_name() == "NVIDIA L40S",
            "Exactly one visible L40S required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(0.005)
    require(torch.cuda.mem_get_info()[0] > 1024 ** 3, "Insufficient free memory for dispatch probe")
    config = read(args.config)["encoder"]
    backbone = AutoConfig.from_pretrained(config["model"], revision=config["revision"],
                                          cache_dir=".cache/torch-models", local_files_only=True)
    backbone._attn_implementation = "sdpa"
    cases = []
    for length in (512, 1536, 3072):
        embedded = torch.zeros(1, length, backbone.hidden_size, dtype=torch.bfloat16)
        padding = torch.ones(1, length, dtype=torch.long)
        padding[:, -17:] = 0
        packed = torch.empty(1, length, 3, backbone.num_attention_heads,
                             backbone.hidden_size // backbone.num_attention_heads,
                             device="cuda", dtype=torch.bfloat16, requires_grad=True)
        query, key, value = [t.transpose(1, 2) for t in packed.unbind(dim=-3)]
        # Rotary operations allocate new query/key tensors; value remains a packed view.
        query, key = query.clone(), key.clone()
        for name, factory in (("global", create_bidirectional_mask),
                              ("local", create_bidirectional_sliding_window_mask)):
            mask = factory(config=backbone, inputs_embeds=embedded, attention_mask=padding)
            require(mask is not None, "Expected explicit padding/local mask")
            mask = mask.to("cuda")
            parameters = SDPAParams(query, key, value, mask, 0.0, False, False)
            selected = torch.ops.aten._fused_sdp_choice(query, key, value, mask, 0.0, False)
            cases.append({"length": length, "kind": name, "batch": 1,
                          "queryStride": list(query.stride()), "valueStride": list(value.stride()),
                          "maskShape": list(mask.shape), "maskDtype": str(mask.dtype),
                          "flashEligible": torch.backends.cuda.can_use_flash_attention(parameters),
                          "efficientEligible": torch.backends.cuda.can_use_efficient_attention(parameters),
                          "cudnnEligible": torch.backends.cuda.can_use_cudnn_attention(parameters),
                          "selected": SDPBackend(selected).name})
            del mask
        del packed, query, key, value, parameters
    peak = torch.cuda.max_memory_allocated()
    require(peak < 200 * 1024 ** 2, "Dispatch probe exceeded its allocation limit")
    report = {"scope": "Metadata dispatch probe on synthetic tensor shapes and actual Transformers ModernBERT mask builders. No encoder weights loaded and no attention forward/backward executed. Not a runtime speed benchmark or a trace of the live training process.",
              "config": args.config, "encoder": config,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(), "deterministic": True,
              "memoryFractionCap": 0.005, "peakAllocatedBytes": peak,
              "scriptHash": digest(Path(__file__).read_bytes()), "cases": cases}
    write(args.output, report)
    print({"cases": cases, "peakAllocatedBytes": peak})


if __name__ == "__main__":
    main()
