#!/usr/bin/env python3
"""
Export Fine-Tuned Whisper to CoreML (WhisperKit format)

Converts a HuggingFace whisper-small checkpoint to CoreML models
compatible with WhisperKit on iOS.

Prerequisites:
    pip install whisperkit coremltools torch transformers

Usage:
    # Using argmaxinc/whisperkittools (recommended)
    python export_coreml.py \
        --model-dir /path/to/finetuned/final \
        --output-dir ./coreml_output \
        --method whisperkittools

    # Using direct coremltools conversion
    python export_coreml.py \
        --model-dir /path/to/finetuned/final \
        --output-dir ./coreml_output \
        --method coremltools

Output structure expected by WhisperKit:
    coreml_output/
    ├── AudioEncoder.mlmodelc/
    ├── MelSpectrogram.mlmodelc/
    ├── TextDecoder.mlmodelc/
    └── config.json
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def convert_via_whisperkittools(model_dir: Path, output_dir: Path):
    """
    Convert using argmaxinc/whisperkittools CLI.
    This is the recommended path as it produces WhisperKit-compatible models.

    Install: pip install whisperkittools
    Repo: https://github.com/argmaxinc/whisperkittools
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Converting via whisperkittools...")
    log.info(f"Input:  {model_dir}")
    log.info(f"Output: {output_dir}")

    # whisperkittools CLI command
    cmd = [
        sys.executable, "-m", "whisperkit.convert",
        "--model-version", str(model_dir),
        "--output-dir", str(output_dir),
        "--model-repo-id", "argmaxinc/whisperkit-coreml",
        # Quantization options (optional):
        # "--encoder-compute-units", "cpuAndNeuralEngine",
        # "--decoder-compute-units", "cpuAndNeuralEngine",
    ]

    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"whisperkittools conversion failed with code {result.returncode}")

    log.info("whisperkittools conversion complete.")
    _verify_output(output_dir)


def convert_via_coremltools(model_dir: Path, output_dir: Path):
    """
    Direct conversion using coremltools + transformers.
    Fallback path if whisperkittools is unavailable.
    """
    try:
        import torch
        import coremltools as ct
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. Install: pip install coremltools transformers torch") from e

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading model from {model_dir}")
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
    model.eval()

    # ── Encoder ─────────────────────────────────────────────────────────────
    log.info("Converting audio encoder...")

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder = model.model.encoder

        def forward(self, input_features):
            return self.encoder(input_features).last_hidden_state

    encoder = EncoderWrapper(model)
    dummy_input = torch.zeros(1, 80, 3000)  # (batch, n_mels, time)
    traced_encoder = torch.jit.trace(encoder, dummy_input)

    encoder_mlmodel = ct.convert(
        traced_encoder,
        inputs=[ct.TensorType(name="input_features", shape=dummy_input.shape)],
        outputs=[ct.TensorType(name="encoder_output_embeds")],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )
    encoder_path = output_dir / "AudioEncoder.mlpackage"
    encoder_mlmodel.save(str(encoder_path))
    log.info(f"Encoder saved: {encoder_path}")

    # ── Text Decoder ─────────────────────────────────────────────────────────
    log.info("Converting text decoder...")
    log.info("Note: Decoder conversion requires stateful handling — consider using whisperkittools for full compatibility.")

    # For a complete decoder conversion (with KV-cache), use whisperkittools.
    # Here we export the decoder without KV-cache as a starting point.
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.decoder = model.model.decoder
            self.proj_out = model.proj_out

        def forward(self, input_ids, encoder_hidden_states):
            out = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
            )
            return self.proj_out(out.last_hidden_state)

    decoder = DecoderWrapper(model)
    dummy_ids = torch.zeros(1, 1, dtype=torch.long)
    dummy_enc = torch.zeros(1, 1500, 512)  # whisper-small encoder dim

    traced_decoder = torch.jit.trace(decoder, (dummy_ids, dummy_enc))
    decoder_mlmodel = ct.convert(
        traced_decoder,
        inputs=[
            ct.TensorType(name="input_ids", shape=dummy_ids.shape, dtype=int),
            ct.TensorType(name="encoder_output_embeds", shape=dummy_enc.shape),
        ],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )
    decoder_path = output_dir / "TextDecoder.mlpackage"
    decoder_mlmodel.save(str(decoder_path))
    log.info(f"Decoder saved: {decoder_path}")

    # ── Save config ──────────────────────────────────────────────────────────
    import json
    processor = WhisperProcessor.from_pretrained(str(model_dir))
    config = {
        "model_type": "whisper-small",
        "language": "arabic",
        "task": "transcribe",
        "vocab_size": model.config.vocab_size,
        "d_model": model.config.d_model,
        "encoder_attention_heads": model.config.encoder_attention_heads,
        "decoder_attention_heads": model.config.decoder_attention_heads,
        "decoder_start_token_id": model.config.decoder_start_token_id,
        "pad_token_id": model.config.pad_token_id,
        "eos_token_id": model.config.eos_token_id,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    log.info("coremltools conversion complete.")
    _verify_output(output_dir)


def _verify_output(output_dir: Path):
    """Check that expected output files exist."""
    expected = ["AudioEncoder.mlpackage", "TextDecoder.mlpackage"]
    missing = [f for f in expected if not (output_dir / f).exists()]
    if missing:
        log.warning(f"Expected files not found: {missing}")
        log.warning("Check conversion output above for errors.")
    else:
        log.info(f"Verification OK. CoreML models at: {output_dir}")
        for p in output_dir.iterdir():
            log.info(f"  {p.name}")


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned Whisper to CoreML")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Path to fine-tuned HuggingFace model directory")
    parser.add_argument("--output-dir", type=Path, default=Path("./coreml_output"),
                        help="Output directory for CoreML models")
    parser.add_argument("--method", choices=["whisperkittools", "coremltools"],
                        default="whisperkittools",
                        help="Conversion method (default: whisperkittools)")
    args = parser.parse_args()

    if not args.model_dir.exists():
        log.error(f"Model directory not found: {args.model_dir}")
        sys.exit(1)

    log.info(f"Conversion method: {args.method}")

    if args.method == "whisperkittools":
        convert_via_whisperkittools(args.model_dir, args.output_dir)
    else:
        convert_via_coremltools(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
