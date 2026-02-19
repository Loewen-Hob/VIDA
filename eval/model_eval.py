import json
import numpy as np
import pandas as pd
import os
import datetime
from sentence_transformers import SentenceTransformer, util

GT_FILE = "evaluation_data_final/test_ground_truth.jsonl"
PRED_FILE = "/ssd1/zhanghongbo04/1216_fanyi/beifeng/VIDA/eval/results/deepseek-vl-7b.jsonl" 
MODEL_PATH = "/home/users/zhanghongbo/all-MiniLM-L6-v2"

CATEGORY_KEYWORDS = {
    "Home Structure": ["structure", "wall", "floor", "window", "layout", "plan", "renovate"],
    "Spatial Information": ["dimension", "size", "area", "measure", "width", "height"],
    "Budget": ["budget", "cost", "price", "spend"],
    "Target Users": ["who", "family", "member", "kid", "child", "pet", "people"],
    "Style Preference": ["style", "modern", "vintage", "color", "vibe", "aesthetic", "feel"],
    "Lifestyle Patterns": ["habit", "hobby", "cook", "work", "lifestyle", "read"],
    "Personalized Needs": ["special", "personal", "custom", "unique"],
    "Storage Requirements": ["storage", "cabinet", "shelf", "closet", "organize"],
    "Inspiration Images": ["inspiration", "reference", "image", "photo", "picture"]
}

def check_hit(text, category):
    if not category: return False
    text = text.lower()
    keywords = CATEGORY_KEYWORDS.get(category, [])
    for kw in keywords:
        if kw in text: return True
    return False

def evaluate():
    base_name = os.path.splitext(os.path.basename(PRED_FILE))[0]
    dir_name = os.path.dirname(PRED_FILE)
    log_file = os.path.join(dir_name, f"{base_name}_report.txt")
    
    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Report for: {base_name}\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*40 + "\n\n")

    log(f"Loading Eval Model: {MODEL_PATH} ...")
    model = SentenceTransformer(MODEL_PATH)
    
    gt_dict = {}
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            gt_dict[str(item['id'])] = item
            
    preds = []
    try:
        with open(PRED_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                preds.append(json.loads(line))
    except FileNotFoundError:
        log(f"Error: File not found {PRED_FILE}")
        return

    scores = {"K-RME": [], "SAS": [], "Max-Sim": [], "Visual_Grounding": []}
    
    log(f"Evaluating {len(preds)} samples...")
    
    for item in preds:
        pid = str(item['id'])
        pred_text = item['prediction']
        
        if pid not in gt_dict: continue
        gt = gt_dict[pid]
        
        l3_cat = gt['annotations'].get('L3', {}).get('missing_elements')
        l2_cat = gt['annotations'].get('L2', {}).get('missing_elements')
        l1_cat = gt['annotations'].get('L1', {}).get('missing_elements')
        
        targets = [c for c in [l3_cat, l2_cat, l1_cat] if c]
        hit_any = any(check_hit(pred_text, t) for t in targets)
        scores["K-RME"].append(1.0 if hit_any else 0.0)
        
        if l3_cat:
            if check_hit(pred_text, l3_cat):
                scores["SAS"].append(1.0)
            else:
                scores["SAS"].append(0.0)
                
        refs = []
        for layer in ['L3', 'L2', 'L1']:
            q = gt['annotations'].get(layer, {}).get('question')
            if q: refs.append(q)
            
        if refs:
            emb_pred = model.encode(pred_text, convert_to_tensor=True)
            emb_refs = model.encode(refs, convert_to_tensor=True)
            scores["Max-Sim"].append(util.cos_sim(emb_pred, emb_refs).max().item())
            
        vis_prompt = gt.get('visual_prompt', '')
        if vis_prompt:
            emb_pred = model.encode(pred_text, convert_to_tensor=True)
            emb_vis = model.encode(vis_prompt, convert_to_tensor=True)
            scores["Visual_Grounding"].append(util.cos_sim(emb_pred, emb_vis).item())

    # --- 输出结果 ---
    log("\n" + "="*40)
    log(f"📊 Evaluation Results: {base_name}")
    log("="*40)
    log(f"1. K-RME (Accuracy):    {np.mean(scores['K-RME']):.2%}")
    log(f"2. SAS (Strategy):      {np.mean(scores['SAS']):.2%}")
    log(f"3. Max-Sim (Semantic):  {np.mean(scores['Max-Sim']):.4f}")
    log(f"4. Visual Grounding:    {np.mean(scores['Visual_Grounding']):.4f}")
    log("="*40)
    print(f"\n[Info] Report saved to: {log_file}")

if __name__ == "__main__":
    evaluate()