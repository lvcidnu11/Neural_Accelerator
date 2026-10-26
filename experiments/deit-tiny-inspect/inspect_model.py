#!/usr/bin/env python3
"""Offline float-model inspection, NOT quantization or a Spike benchmark."""
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import timm
import torch

ROOT = Path(__file__).resolve().parent


def main():
    results = ROOT / "results"
    manifest = json.loads((results / "model-manifest.json").read_text())
    checkpoint = ROOT / manifest["checkpoint_file"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest["checkpoint_sha256"]
    torch.manual_seed(6455)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = timm.create_model(manifest["model"], pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    # Use explicit attention operations for transparent inspection.
    for block in model.blocks:
        if hasattr(block.attn, "fused_attn"):
            block.attn.fused_attn = False

    records = []
    captured = {}
    handles = []
    selected = {"blocks.0.attn.qkv", "blocks.0.attn.proj",
                "blocks.0.mlp.fc1", "blocks.0.mlp.fc2"}

    def hook_for(name):
        def hook(module, inputs, output):
            x = inputs[0]
            row = dict(name=name, kind=type(module).__name__,
                       input_shape=list(x.shape), output_shape=list(output.shape))
            if isinstance(module, torch.nn.Linear):
                m, k, n = math.prod(x.shape[:-1]), module.in_features, module.out_features
                row.update(M=m, K=k, N=n, macs=m*k*n)
            elif isinstance(module, torch.nn.Conv2d):
                row["macs"] = output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            else:
                row["macs"] = None
            records.append(row)
            if name in selected:
                captured[name] = dict(input=x.detach().clone(),
                    weight=module.weight.detach().clone(),
                    bias=module.bias.detach().clone() if module.bias is not None else None,
                    output=output.detach().clone())
        return hook

    wanted = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm,
              torch.nn.GELU, torch.nn.Softmax)
    for name, module in model.named_modules():
        if isinstance(module, wanted):
            handles.append(module.register_forward_hook(hook_for(name)))

    # Synthetic, already model-space input: NOT an ImageNet image or calibration set.
    x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    with torch.inference_mode():
        y = model(x)
    for handle in handles:
        handle.remove()
    assert list(y.shape) == [1, 1000] and bool(torch.isfinite(y).all())
    assert len(model.blocks) == 12 and model.embed_dim == 192
    assert len(captured) == 4
    qkv = next(row for row in records if row["name"] == "blocks.0.attn.qkv")
    assert qkv["input_shape"] == [1, 197, 192]

    # Explicit attention products are functional matmuls, not nn.Linear hooks.
    # Counts below are derived from observed tokens and configured head dimensions.
    tokens = qkv["input_shape"][1]
    attention = []
    for i, block in enumerate(model.blocks):
        heads = block.attn.num_heads
        head_dim = model.embed_dim // heads
        for op in ["Q_times_K_transpose", "probabilities_times_V"]:
            attention.append(dict(block=i, op=op, heads=heads,
                tokens=tokens, head_dim=head_dim,
                macs=heads*tokens*tokens*head_dim,
                method="shape-derived, not retired-instruction or cycle measurement"))

    linear_macs = sum(r["macs"] for r in records if r["kind"] == "Linear")
    conv_macs = sum(r["macs"] for r in records if r["kind"] == "Conv2d")
    attention_macs = sum(r["macs"] for r in attention)
    with torch.inference_mode():
        for fixture in captured.values():
            independent = fixture["input"] @ fixture["weight"].T
            if fixture["bias"] is not None:
                independent = independent + fixture["bias"]
            # Floating arithmetic ordering may vary; this is not bit-exact INT8 validation.
            torch.testing.assert_close(independent, fixture["output"], rtol=1e-4, atol=1e-5)
    torch.save(dict(seed=6455, input=x, logits=y, operators=captured,
                    quantized=False, calibration_data=False), results / "float-fixtures.pt")

    report = dict(model=manifest["model"], quantized=False,
        checkpoint_sha256=manifest["checkpoint_sha256"], seed=6455,
        input_kind="synthetic normal random tensor; no dataset accuracy measured",
        input_shape=list(x.shape), output_shape=list(y.shape),
        parameter_count=sum(p.numel() for p in model.parameters()),
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        blocks=len(model.blocks), tokens=tokens, embedding_dim=model.embed_dim,
        heads=model.blocks[0].attn.num_heads,
        mlp_hidden_dim=model.blocks[0].mlp.fc1.out_features,
        linear_macs=linear_macs, patch_conv_macs=conv_macs,
        attention_matmul_macs=attention_macs,
        counted_macs=linear_macs+conv_macs+attention_macs,
        mac_definition="One multiplication and its accumulation = one MAC. Shape-derived; excludes normalization, softmax, GELU, bias, residual additions and other non-MAC operations.",
        operator_records=records, attention_records=attention,
        versions=dict(python=sys.version, torch=torch.__version__, timm=timm.__version__,
                      platform=platform.platform()),
        data_config=model.pretrained_cfg)
    (results / "inspection.json").write_text(json.dumps(report, indent=2, default=str)+"\n")
    rows = [r for r in records if r["kind"] == "Linear"]
    with (results / "linear-shapes.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["name","M","K","N","macs"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k:r[k] for k in writer.fieldnames})
    (results / "requirements-lock.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "--no-cache-dir", "freeze"], text=True))
    (results / "source-sha256.json").write_text(json.dumps({
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ["inspect_model.py","download_weights.py","requirements.txt","run.sh"]}, indent=2)+"\n")
    print(f"PASS: strict official checkpoint load; finite output {list(y.shape)}")
    print(f"Parameters: {report['parameter_count']:,}; blocks={report['blocks']}; tokens={tokens}")
    for row in rows:
        if row["name"] in selected:
            print(f"{row['name']}: M={row['M']} K={row['K']} N={row['N']}; MAC={row['macs']:,}")
    print(f"Counted MACs/inference (shape-derived): {report['counted_macs']:,}")
    print("Saved FLOAT fixtures. No W8A8 conversion, accuracy evaluation or Spike model execution yet.")


if __name__ == "__main__":
    main()
