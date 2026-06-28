#!/usr/bin/env python3
"""Standalone SmolVLM A_command parser.

This script intentionally imports no ROS modules.  It loads a local VLM,
reads one cropped/warped A_command image, and prints validated JSON only to
stdout on success.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
from typing import Dict, Optional


PART_CLASS_NAMES = [
    "flange_nut",
    "gear_ring",
    "spacer_ring",
    "hex_nut",
    "dome_nut",
]
VALID_COUNTS = (-1, 0, 1, 2, 3, 4, 5)
CLASS_ALIASES = {
    "flange_nut": "flange_nut",
    "flange nut": "flange_nut",
    "flangenut": "flange_nut",
    "플랜지 너트": "flange_nut",
    "플랜지너트": "flange_nut",
    "gear_ring": "gear_ring",
    "gear ring": "gear_ring",
    "gearring": "gear_ring",
    "기어 링": "gear_ring",
    "기어링": "gear_ring",
    "spacer_ring": "spacer_ring",
    "spacer ring": "spacer_ring",
    "spacerring": "spacer_ring",
    "스페이서 링": "spacer_ring",
    "스페이서링": "spacer_ring",
    "hex_nut": "hex_nut",
    "hex nut": "hex_nut",
    "hexnut": "hex_nut",
    "육각 너트": "hex_nut",
    "육각너트": "hex_nut",
    "dome_nut": "dome_nut",
    "dome nut": "dome_nut",
    "domenut": "dome_nut",
    "돔 너트": "dome_nut",
    "돔너트": "dome_nut",
}

PROMPT = """Read ONLY the visible cropped/warped A-command parts table image.

Output exactly one JSON object and nothing else.
No explanation. No Markdown. No code block.

The JSON object must have exactly these 5 keys:
flange_nut, gear_ring, spacer_ring, hex_nut, dome_nut

Each value must be an integer from 0 to 5.
If a count is unclear, hidden, cropped, or unreadable, use -1.
Do not use the row order to choose keys; match the displayed part label/icon to the key.

{
  "flange_nut": -1,
  "gear_ring": -1,
  "spacer_ring": -1,
  "hex_nut": -1,
  "dome_nut": -1
}
"""


def _eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local SmolVLM for A_command JSON.")
    parser.add_argument("--model", required=True, help="Local SmolVLM model directory")
    parser.add_argument("--image", required=True, help="Cropped or warped A_command image")
    parser.add_argument("--timeout", type=float, default=10.0, help="Generation timeout hint")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def _extract_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    source = text.strip()
    if not source:
        raise ValueError("empty_model_output")
    for idx, char in enumerate(source):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(source[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no_json_object")


def _validate_json(parsed: dict) -> Dict[str, int]:
    if not isinstance(parsed, dict):
        raise ValueError("top_level_not_object")
    expected = set(PART_CLASS_NAMES)
    normalized: Dict[str, int] = {}
    for key, value in parsed.items():
        class_name = _normalize_class_key(key)
        if not class_name:
            continue
        if isinstance(value, bool):
            raise ValueError(f"{class_name}_not_int")
        if isinstance(value, str):
            value = value.strip()
            if not re.fullmatch(r"-?\d+", value):
                raise ValueError(f"{class_name}_not_int")
            value = int(value)
        if not isinstance(value, int):
            raise ValueError(f"{class_name}_not_int")
        normalized[class_name] = int(value)
    actual = set(normalized.keys())
    if actual != expected:
        raise ValueError(f"keys_mismatch:{sorted(actual)}")
    out = {}
    for name in PART_CLASS_NAMES:
        value = normalized[name]
        if value not in VALID_COUNTS:
            raise ValueError(f"{name}_out_of_range:{value}")
        out[name] = int(value)
    return out


def _normalize_class_key(key) -> Optional[str]:
    raw = str(key).strip()
    normalized = raw.lower().replace("-", " ").replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    compact = normalized.replace(" ", "")
    return CLASS_ALIASES.get(raw) or CLASS_ALIASES.get(normalized) or CLASS_ALIASES.get(compact)


def _timeout_handler(_signum, _frame):
    raise TimeoutError("vlm_generation_timeout")


def _load_model(model_path: str):
    try:
        import torch
        from transformers import AutoProcessor
    except Exception as exc:
        _eprint(f"dependency import failed: {exc}")
        return None, None, None, 2

    model_classes = []
    try:
        from transformers import AutoModelForImageTextToText

        model_classes.append(AutoModelForImageTextToText)
    except Exception:
        pass
    try:
        from transformers import AutoModelForVision2Seq

        model_classes.append(AutoModelForVision2Seq)
    except Exception:
        pass
    try:
        from transformers import AutoModelForCausalLM

        model_classes.append(AutoModelForCausalLM)
    except Exception:
        pass

    if not model_classes:
        _eprint("no compatible transformers AutoModel class is available")
        return None, None, None, 2

    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    except Exception as exc:
        _eprint(f"processor load failed: {exc}")
        return None, None, None, 3

    cuda = torch.cuda.is_available()
    if cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        dtype = torch.bfloat16
    elif cuda:
        dtype = torch.float16
    else:
        dtype = torch.float32

    last_error = None
    for model_class in model_classes:
        kwargs = {
            "local_files_only": True,
            "torch_dtype": dtype,
        }
        if cuda:
            kwargs["device_map"] = "auto"
        try:
            model = model_class.from_pretrained(model_path, **kwargs)
            if not cuda:
                model = model.to("cpu")
            model.eval()
            return processor, model, torch, 0
        except Exception as exc:
            last_error = exc
            _eprint(f"{model_class.__name__} load failed: {exc}")
    _eprint(f"model load failed: {last_error}")
    return None, None, None, 3


def _build_inputs(processor, image, torch):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    }]
    try:
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        prompt = PROMPT

    try:
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
    except Exception:
        inputs = processor(text=prompt, images=image, return_tensors="pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def main() -> int:
    args = _parse_args()
    if not os.path.isdir(args.model):
        _eprint(f"model directory not found: {args.model}")
        return 2
    if not os.path.exists(args.image):
        _eprint(f"image not found: {args.image}")
        return 2

    try:
        from PIL import Image
    except Exception as exc:
        _eprint(f"pillow import failed: {exc}")
        return 2

    processor, model, torch, code = _load_model(args.model)
    if code != 0:
        return code

    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as exc:
        _eprint(f"image load failed: {exc}")
        return 2

    try:
        inputs = _build_inputs(processor, image, torch)
        generation_kwargs = {
            "max_new_tokens": max(1, int(args.max_new_tokens)),
            "do_sample": args.temperature > 0.0,
        }
        if args.temperature > 0.0:
            generation_kwargs["temperature"] = float(args.temperature)

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, max(1.0, float(args.timeout)))
        try:
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)

        input_len = 0
        if "input_ids" in inputs and hasattr(inputs["input_ids"], "shape"):
            input_len = int(inputs["input_ids"].shape[-1])
        if hasattr(generated, "__getitem__"):
            generated_ids = generated[:, input_len:] if input_len > 0 else generated
        else:
            generated_ids = generated
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except TimeoutError as exc:
        _eprint(str(exc))
        return 5
    except Exception as exc:
        _eprint(f"generation failed: {exc}")
        return 5

    try:
        clean = _validate_json(_extract_json_object(text))
    except Exception as exc:
        _eprint(f"model output validation failed: {exc}")
        _eprint(f"raw model output: {text}")
        return 4

    print(json.dumps(clean, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
