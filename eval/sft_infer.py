import os
import json
import torch
from tqdm import tqdm
from swift.llm import PtEngine, RequestConfig, InferRequest
MODEL_DIR = "/ssd1/zhanghongbo04/1216_fanyi/beifeng/VIDA/deepseek-vl-7b"

OUTPUT_FILE = "results/deepseek-vl-7b.jsonl"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

INPUT_FILE = "evaluation_data_final/test_inputs_with_captions.jsonl"

# =========================================
sys = '''
You are a professional interior designer. Based on the images and requirements, identify the user's latent needs and formulate appropriate questions.
Please understand which element should be the primary focus of your inquiry, then pose only one question at a time. Avoid asking multiple questions simultaneously. Keep your word count within the limits of a single question, ensuring it remains concise and not overly complex.
'''
def main():
    print(f"正在加载模型: {MODEL_DIR} ...")
    
    engine = PtEngine(MODEL_DIR, max_batch_size=1) 

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"共加载 {len(test_data)} 条测试数据，开始推理...")
    
    infer_requests = []
    for item in test_data:
        query = item['prompt'].replace('<image>', '').strip()
        image_path = item['image']
        image_caption = item['image_description']
        caption_input = f'{query},this image description is {image_caption}'
        messages = [
            {'role': 'system', 'content': sys},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': image_path},
                {'type': 'text', 'text': query},
            ]}
        ]
        infer_requests.append(InferRequest(messages=messages))

    request_config = RequestConfig(
        max_tokens=128, 
        temperature=0.0, 
        top_p=0.9
    )

    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i, req in enumerate(tqdm(infer_requests)):
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
                print(f"\n[Error] ID {test_data[i]['id']} 推理失败: {e}")
                json.dump({"id": test_data[i]['id'], "prediction": "Error"}, f_out, ensure_ascii=False)
                f_out.write('\n')

    print(f"\n推理完成！结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()