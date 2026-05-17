"""Reward function used for VIDA RL training.

The reward can be imported by training code directly or registered as an
external ModelScope Swift ORM plugin.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Iterable

from vida.constants import CATEGORY_KEYWORDS

DEFAULT_REWARD_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def check_hit(text: str, category: str | None) -> bool:
    """Return whether text mentions keywords for a missing intent category."""
    if not category:
        return False
    text = text.lower()
    return any(keyword in text for keyword in CATEGORY_KEYWORDS.get(category, []))


@lru_cache(maxsize=2)
def get_reward_model(model_name_or_path: str | None = None):
    from sentence_transformers import SentenceTransformer

    model_name_or_path = (
        model_name_or_path
        or os.environ.get("VIDA_REWARD_MODEL")
        or DEFAULT_REWARD_MODEL
    )
    return SentenceTransformer(model_name_or_path, device=os.environ.get("VIDA_REWARD_DEVICE", "cpu"))


def _load_label(label: Any) -> dict[str, Any]:
    if isinstance(label, str):
        return json.loads(label)
    if isinstance(label, dict):
        return label
    raise TypeError(f"Unsupported label type: {type(label)!r}")


def _missing_category(gt: dict[str, Any], level: str) -> str | None:
    direct_key = f"{level}_cat"
    if direct_key in gt:
        return gt.get(direct_key)
    annotations = gt.get("annotations", {})
    level_data = annotations.get(level, {})
    return level_data.get("missing_elements")


def _reference_questions(gt: dict[str, Any]) -> list[str]:
    questions = gt.get("questions")
    if questions:
        return [q for q in questions if q]

    refs: list[str] = []
    for level in ("L3", "L2", "L1"):
        q = gt.get("annotations", {}).get(level, {}).get("question")
        if q:
            refs.append(q)
    return refs


def semantic_similarity_reward(text: str, refs: Iterable[str], encoder) -> float:
    from sentence_transformers import util

    refs = [ref for ref in refs if ref]
    if not refs:
        return 0.0
    emb_gen = encoder.encode(text, convert_to_tensor=True)
    emb_refs = encoder.encode(refs, convert_to_tensor=True)
    return max(0.0, float(util.cos_sim(emb_gen, emb_refs).max().item()))


def hierarchy_reward(text: str, gt: dict[str, Any]) -> float:
    if check_hit(text, _missing_category(gt, "L3")):
        return 1.0
    if check_hit(text, _missing_category(gt, "L2")):
        return 0.6
    if check_hit(text, _missing_category(gt, "L1")):
        return 0.3
    return 0.0


def lexical_visual_reward(text: str, visual_prompt: str) -> float:
    """Lightweight visual grounding reward used in the released experiments."""
    if not visual_prompt:
        return 0.0
    visual_words = {w for w in re.findall(r"\w+", visual_prompt.lower()) if len(w) > 3}
    generated_words = set(re.findall(r"\w+", text.lower()))
    return min(len(generated_words & visual_words) * 0.2, 1.0)


def embedding_visual_reward(
    text: str,
    visual_prompt: str,
    encoder,
    low: float = 0.25,
    high: float = 0.70,
) -> float:
    from sentence_transformers import util

    """Semantic visual grounding reward based on visual-fact similarity."""
    if not visual_prompt:
        return 0.0

    visual_units = [
        unit.strip()
        for unit in re.split(r"[.;,\n]", visual_prompt)
        if len(unit.strip()) > 8
    ]
    if not visual_units:
        visual_units = [visual_prompt]

    emb_gen = encoder.encode(text, convert_to_tensor=True)
    emb_vis = encoder.encode(visual_units, convert_to_tensor=True)
    sim = float(util.cos_sim(emb_gen, emb_vis).max().item())
    return max(0.0, min((sim - low) / (high - low), 1.0))


def format_reward(text: str) -> float:
    score = 0.1 if "?" in text or "？" in text else -0.5
    if len(text) < 10:
        score -= 0.5
    if len(text) > 100:
        score -= 0.2
    return score


def vida_reward(
    text: str,
    label: Any,
    encoder: Any | None = None,
    visual_mode: str = "lexical",
    weights: tuple[float, float, float, float] = (2.0, 1.0, 0.5, 0.5),
) -> float:
    """Compute VIDA's composite reward.

    Weight order is ``(hierarchy, semantic_similarity, visual_grounding, format)``.
    """
    gt = _load_label(label)
    encoder = encoder or get_reward_model()

    h_score = hierarchy_reward(text, gt)
    sim_score = semantic_similarity_reward(text, _reference_questions(gt), encoder)
    visual_prompt = gt.get("visual_prompt") or gt.get("image_description") or ""
    if visual_mode == "embedding":
        v_score = embedding_visual_reward(text, visual_prompt, encoder)
    elif visual_mode == "lexical":
        v_score = lexical_visual_reward(text, visual_prompt)
    else:
        raise ValueError(f"Unknown visual reward mode: {visual_mode}")
    f_score = format_reward(text)

    h_weight, sim_weight, v_weight, f_weight = weights
    return float(
        h_weight * h_score
        + sim_weight * sim_score
        + v_weight * v_score
        + f_weight * f_score
    )


class VidaGSPOReward:
    """Swift ORM-compatible reward class."""

    def __call__(self, completions: list[str], **kwargs: Any) -> list[float]:
        labels = kwargs.get("label") or kwargs.get("ground_truth") or kwargs.get("target")
        if labels is None:
            return [0.0] * len(completions)

        visual_mode = os.environ.get("VIDA_VISUAL_REWARD", "lexical")
        encoder = get_reward_model()
        rewards: list[float] = []
        for text, label in zip(completions, labels):
            try:
                rewards.append(vida_reward(text, label, encoder=encoder, visual_mode=visual_mode))
            except Exception:
                rewards.append(0.0)
        return rewards


def _register_swift_plugin() -> None:
    try:
        from swift.plugin import ORM, orms
    except Exception:
        return

    class SwiftVidaGSPOReward(VidaGSPOReward, ORM):
        pass

    orms["vida_gspo_reward"] = SwiftVidaGSPOReward


_register_swift_plugin()
