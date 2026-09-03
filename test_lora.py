import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


MODEL_PATH = os.getenv("PCB_BASE_MODEL_PATH", "models/Qwen2.5-7B-Instruct")


# 1. 4-bit QLoRA 配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# 2. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)


# 3. 加载 4-bit 基础模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)


# 4. 为 k-bit 训练做准备
model = prepare_model_for_kbit_training(model)


# 5. LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)


# 6. 把 LoRA 挂到 Qwen 上
model = get_peft_model(
    model,
    lora_config
)


# 7. 打印可训练参数
model.print_trainable_parameters()


# 8. 显存占用
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

print(f"已分配显存: {allocated:.2f} GB")
print(f"已保留显存: {reserved:.2f} GB")

print("\n=== LoRA layers ===")

count = 0

for name, module in model.named_modules():
    if "lora_A" in name or "lora_B" in name:
        print(name)
        count += 1

        if count >= 20:
            break