"""Run VIDA inference with ModelScope Swift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from vida.constants import SYSTEM_PROMPT


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model path or hub id.")
    parser.add_argument("--adapter", default=None, help="Optional LoRA/RL checkpoint.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    import torch
    from swift.llm import InferRequest, PtEngine, RequestConfig

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]

    engine_kwargs = {"model_id_or_path": args.model, "max_batch_size": 1, "torch_dtype": dtype}
    if args.adapter:
        engine_kwargs["adapters"] = [args.adapter]
    engine = PtEngine(**engine_kwargs)

    request_config = RequestConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    data = read_jsonl(args.input_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f_out:
        for item in tqdm(data, desc="Inference"):
            query = item.get("prompt") or item.get("user_request") or ""
            query = query.replace("<image>", "").strip()
            image_path = item.get("image")
            if not image_path:
                raise ValueError(f"Missing image path for sample id={item.get('id')}")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": query},
                    ],
                },
            ]
            request = InferRequest(messages=messages)
            try:
                response = engine.infer([request], request_config)[0].choices[0].message.content
            except Exception as exc:
                response = f"Error: {exc}"

            json.dump({"id": item.get("id"), "prediction": response}, f_out, ensure_ascii=False)
            f_out.write("\n")
            f_out.flush()


if __name__ == "__main__":
    main()
