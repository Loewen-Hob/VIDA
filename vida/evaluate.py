"""Evaluate VIDA predictions against InteriorClarify-style annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vida.reward import check_hit


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def missing_category(gt: dict, level: str) -> str | None:
    return gt.get("annotations", {}).get(level, {}).get("missing_elements")


def reference_questions(gt: dict) -> list[str]:
    refs = []
    for level in ("L3", "L2", "L1"):
        question = gt.get("annotations", {}).get(level, {}).get("question")
        if question:
            refs.append(question)
    return refs


def evaluate(ground_truth_file: Path, prediction_file: Path, model_name_or_path: str) -> dict[str, float]:
    from sentence_transformers import SentenceTransformer, util

    encoder = SentenceTransformer(model_name_or_path)
    gt_by_id = {str(item["id"]): item for item in read_jsonl(ground_truth_file)}

    scores = {
        "K-RME": [],
        "SAS": [],
        "Max-Sim": [],
        "V-G": [],
    }

    for item in read_jsonl(prediction_file):
        pid = str(item["id"])
        if pid not in gt_by_id:
            continue

        prediction = item.get("prediction", "")
        gt = gt_by_id[pid]
        categories = [missing_category(gt, level) for level in ("L3", "L2", "L1")]
        categories = [category for category in categories if category]

        scores["K-RME"].append(float(any(check_hit(prediction, category) for category in categories)))

        l3_category = missing_category(gt, "L3")
        if l3_category:
            scores["SAS"].append(float(check_hit(prediction, l3_category)))

        refs = reference_questions(gt)
        if refs:
            emb_pred = encoder.encode(prediction, convert_to_tensor=True)
            emb_refs = encoder.encode(refs, convert_to_tensor=True)
            scores["Max-Sim"].append(float(util.cos_sim(emb_pred, emb_refs).max().item()))

        visual_prompt = gt.get("visual_prompt", "")
        if visual_prompt:
            emb_pred = encoder.encode(prediction, convert_to_tensor=True)
            emb_vis = encoder.encode(visual_prompt, convert_to_tensor=True)
            scores["V-G"].append(float(util.cos_sim(emb_pred, emb_vis).item()))

    return {name: float(np.mean(values)) if values else 0.0 for name, values in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-file", type=Path, required=True)
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    results = evaluate(args.ground_truth_file, args.prediction_file, args.embedding_model)
    print(f"K-RME:   {results['K-RME']:.2%}")
    print(f"SAS:     {results['SAS']:.2%}")
    print(f"Max-Sim: {results['Max-Sim']:.4f}")
    print(f"V-G:     {results['V-G']:.4f}")


if __name__ == "__main__":
    main()
