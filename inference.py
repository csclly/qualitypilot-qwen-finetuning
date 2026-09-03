import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


MODEL_PATH = os.getenv("PCB_BASE_MODEL_PATH", "models/Qwen2.5-7B-Instruct")
LORA_PATH = os.getenv("PCB_LORA_PATH", "output/pcb-lora-run")


# ============================================================
# 1. 4-bit 量化配置
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 2. tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)


# ============================================================
# 3. 推理函数
# ============================================================

def generate_answer(model, question):

    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.05,
        )

    # 只取新生成的部分
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return answer


# ============================================================
# 4. 测试问题
# ============================================================

question = "PCB出现开路缺陷，常见原因有哪些？"


# ============================================================
# 5. 加载 Base Qwen
# ============================================================

print("\n==============================")
print("正在加载 Base Qwen...")
print("==============================")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

base_model.eval()


print("\n==============================")
print("Base Qwen 回答")
print("==============================")

base_answer = generate_answer(
    base_model,
    question
)

print(base_answer)


# ============================================================
# 6. 在 Base Qwen 上加载 LoRA
# ============================================================

print("\n==============================")
print("正在加载 PCB LoRA...")
print("==============================")

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)

lora_model.eval()


print("\n==============================")
print("Qwen + LoRA 回答")
print("==============================")

lora_answer = generate_answer(
    lora_model,
    question
)

print(lora_answer)