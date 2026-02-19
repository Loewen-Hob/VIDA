import json
import re
from typing import List
from sentence_transformers import SentenceTransformer, util
from swift.plugin import ORM, orms

_REWARD_MODEL = None

def get_reward_model():
    global _REWARD_MODEL
    if _REWARD_MODEL is None:
        model_path = '/root/paddlejob/workspace/env_run/output/VIDA/all-MiniLM-L6-v2'
        print(f">> [Plugin] Loading SentenceTransformer from {model_path} ...")

        _REWARD_MODEL = SentenceTransformer(model_path, device='cpu')
    return _REWARD_MODEL

CATEGORY_KEYWORDS = {
    "Home Structure": ["structure", "wall", "floor", "window", "layout", "renovate", "demolish"],
    "Spatial Information": ["dimension", "size", "area", "measure", "width", "height", "scale"],
    "Budget": ["budget", "cost", "price", "spend", "range"],
    "Target Users": ["who", "family", "member", "kid", "child", "pet", "people"],
    "Style Preference": ["style", "modern", "vintage", "color", "tone", "vibe", "aesthetic"],
    "Lifestyle Patterns": ["habit", "hobby", "cook", "work", "lifestyle", "read", "game"],
    "Personalized Needs": ["special", "personal", "custom", "unique", "smart"],
    "Storage Requirements": ["storage", "cabinet", "shelf", "closet", "organize", "space"],
    "Inspiration Images": ["inspiration", "reference", "image", "photo", "picture"]
}

def check_hit(text, category):
    if not category: return False
    text = text.lower()
    keywords = CATEGORY_KEYWORDS.get(category, [])
    for kw in keywords:
        if kw in text: return True
    return False

class VidaGSPORewardV2(ORM):
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        encoder = get_reward_model()
        
        labels = kwargs.get('label')
        if labels is None:
             labels = kwargs.get('ground_truth', [])

        for text, label_str in zip(completions, labels):
            total_score = 0.0
            try:
                gt = json.loads(label_str) if isinstance(label_str, str) else label_str
            except Exception as e:
                rewards.append(0.0)
                continue
                
            text_lower = text.lower()
            
            valid_questions = [q for q in gt.get('questions', []) if q]
            sim_score = 0.0
            if valid_questions:
                emb_gen = encoder.encode(text, convert_to_tensor=True)
                emb_refs = encoder.encode(valid_questions, convert_to_tensor=True)
                sim_score = util.cos_sim(emb_gen, emb_refs).max().item()
                sim_score = max(0, sim_score)
            total_score += 1.0 * sim_score 

            h_score = 0.0
            if gt.get('L3_cat') and check_hit(text, gt['L3_cat']): h_score = 1.0
            elif gt.get('L2_cat') and check_hit(text, gt['L2_cat']): h_score = 0.6
            elif gt.get('L1_cat') and check_hit(text, gt['L1_cat']): h_score = 0.3
            total_score += 2.0 * h_score 

            v_score = 0.0
            vis_prompt = gt.get('visual_prompt', '').lower()
            if vis_prompt:
                vis_words = {w for w in re.findall(r'\w+', vis_prompt) if len(w) > 3}
                gen_words = set(re.findall(r'\w+', text_lower))
                overlap = [w for w in gen_words if w in vis_words]
                v_score = min(len(overlap) * 0.2, 1.0)
            total_score += 0.5 * v_score 

            f_score = 0.0
            if '?' in text or '？' in text: f_score += 0.1
            else: f_score -= 0.5
            if len(text) < 10: f_score -= 0.5
            if len(text) > 100: f_score -= 0.2
            total_score += 0.5 * f_score 
            
            rewards.append(float(total_score))
            
        return rewards

orms['vida_reward_v2'] = VidaGSPORewardV2
print(">> [Plugin V2] Loaded and registered 'vida_reward_v2'")