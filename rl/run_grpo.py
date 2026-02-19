import os
from swift.llm import rlhf_main, RLHFArguments
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_PATH = os.path.join(CURRENT_DIR, "my_plugin_v2.py") 

if __name__ == '__main__':
    args = RLHFArguments(
        rlhf_type='grpo',
        model_type='qwen3_vl',
        model="Qwen3-VL-8B-Instruct",
        
        dataset=["/datasets/MS-S-train_rl.jsonl"],
        
        output_dir="output/vida_gspo_rl",
        
        external_plugins=[PLUGIN_PATH], 
        reward_funcs=['vida_reward_v2'], 
        # ----------------
        
        train_type='lora',
        num_train_epochs=1,
        num_generations=4, 
        max_completion_length=128,
        beta=0.01,
        lora_rank=64,
        lora_alpha=128,
        target_modules=['all-linear'],
        freeze_vit=True,
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=1,
        save_steps=50,
        save_only_model=True,
        report_to=['tensorboard'],
        system="system_prompt.txt",
        # deepspeed='zero2'
        deepspeed=None
    )

    print(f">> [Main] External plugin path: {PLUGIN_PATH}")
    rlhf_main(args)