import os
import json
import torch
from tqdm import tqdm
from swift.llm import PtEngine, RequestConfig, InferRequest

BASE_MODEL_DIR = "/home/users/zhanghongbo/Qwen3-VL-8B-Instruct"
RL_CHECKPOINT = "/sft/output/qwen3_vl_8b_lora_sft_stable/v3-20251228-102420/checkpoint-144"

INPUT_FILE = "evaluation_data_final/test_inputs.jsonl"

OUTPUT_FILE = "results/sft_128_output.jsonl"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

SYSTEM_PROMPT = """
You are a professional interior designer. Based on the images and requirements, identify the user's latent needs and formulate appropriate questions.
Please understand which element should be the primary focus of your inquiry, then pose only one question at a time. Avoid asking multiple questions simultaneously. Keep your word count within the limits of a single question, ensuring it remains concise and not overly complex.
"""

# =========================================

def main():
    print(f"正在加载 Base 模型: {BASE_MODEL_DIR}")
    print(f"正在挂载 RL LoRA: {RL_CHECKPOINT}")
    
    engine = PtEngine(
        model_id_or_path=BASE_MODEL_DIR,
        adapters=[RL_CHECKPOINT], 
        max_batch_size=1,
        torch_dtype=torch.bfloat16
    )
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"加载测试集: {len(test_data)} 条")
    print(f"结果将保存至: {OUTPUT_FILE}")
    
    infer_requests = []
    for item in test_data:
        query = item['prompt'].replace('<image>', '').strip()
        image_path = item['image']
        
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': image_path},
                {'type': 'text', 'text': query}
            ]}
        ]
        infer_requests.append(InferRequest(messages=messages))

    request_config = RequestConfig(
        max_tokens=128, 
        temperature=0.0, 
        top_p=0.9
    )
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i, req in enumerate(tqdm(infer_requests, desc="Inference")):
            try:
                resp_list = engine.infer([req], request_config)
                response = resp_list[0].choices[0].message.content
                
                result_entry = {
                    "id": test_data[i]['id'],
                    "prediction": response
                }
                json.dump(result_entry, f_out, ensure_ascii=False)
                f_out.write('\n')
                f_out.flush()
                
            except Exception as e:
                print(f"\n[Error] ID {test_data[i]['id']} Failed: {e}")
                json.dump({"id": test_data[i]['id'], "prediction": "Error"}, f_out, ensure_ascii=False)
                f_out.write('\n')

    print(f"\n✅ 推理完成！")

if __name__ == "__main__":
    main()