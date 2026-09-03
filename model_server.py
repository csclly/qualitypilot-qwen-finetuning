import os
import time
import uuid
import threading

import torch

from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# 1. 基础配置
# ============================================================

MODEL_PATH = os.getenv("PCB_BASE_MODEL_PATH", "models/Qwen2.5-7B-Instruct")

LORA_PATH = os.getenv("PCB_LORA_PATH", "output/pcb-lora-run")

MODEL_NAME = "pcb-qwen-lora"

# 可以通过环境变量设置
API_KEY = os.getenv("PCB_LLM_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("必须设置非空 PCB_LLM_API_KEY 后再启动服务")


# ============================================================
# 2. 工业场景系统提示词
# ============================================================

GROUNDING_SYSTEM_PROMPT = """
你是一个面向PCB制造与AOI质量检测场景的工业质量异常分析助手。

你必须遵循以下规则：

1. 只能基于用户提供的信息、知识库检索结果以及工具返回的数据进行分析。

2. 不得自行虚构以下信息：
   - 机台编号
   - 批次编号
   - 操作人员
   - 工艺参数
   - 阈值
   - 检测结果
   - 材料批次
   - MES/QMS查询结果

3. 当证据不足时，必须明确说明：
   “当前证据不足，无法确定唯一根因。”
   并说明需要补充哪些信息。

4. 候选根因只能作为待验证假设。
   在没有足够证据的情况下，不得描述为已经确认的根因。

5. 工业异常分析优先考虑证据链：
   AOI原图/人工复核
   → Gerber或设计基准
   → MES工艺和批次信息
   → QMS历史案例
   → 电测/终检结果。

6. 涉及以下高风险动作：
   - 停线
   - 修改设备参数
   - 修改工艺参数
   - 批量报废
   - 正式创建处置工单

   只能提供建议，不得直接执行，
   必须经过人工审批。

7. 回答应尽量结构化、工程化、可执行，
   避免空泛的通用描述。
""".strip()


# ============================================================
# 3. 路径检查
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"基础模型不存在：{MODEL_PATH}"
    )

if not os.path.exists(LORA_PATH):
    raise FileNotFoundError(
        f"LoRA Adapter不存在：{LORA_PATH}"
    )


# ============================================================
# 4. GPU检查
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "没有检测到CUDA GPU"
    )

print("=" * 70)
print("PCB Qwen LoRA Model Server")
print("=" * 70)

print("GPU:", torch.cuda.get_device_name(0))
print("Base Model:", MODEL_PATH)
print("LoRA:", LORA_PATH)


# ============================================================
# 5. 4-bit配置
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 6. Tokenizer
# ============================================================

print("\n正在加载Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


# ============================================================
# 7. Base Qwen
# ============================================================

print("\n正在加载Base Qwen2.5-7B...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,

    quantization_config=bnb_config,

    device_map="auto",

    trust_remote_code=True,
)

base_model.eval()


# ============================================================
# 8. LoRA
# ============================================================

print("\n正在加载PCB LoRA Adapter...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)

model.eval()

# 推理允许使用KV Cache
model.config.use_cache = True


print("\n模型加载完成")

print(
    "GPU allocated:",
    round(
        torch.cuda.memory_allocated()
        / 1024**3,
        2,
    ),
    "GB",
)

print(
    "GPU reserved:",
    round(
        torch.cuda.memory_reserved()
        / 1024**3,
        2,
    ),
    "GB",
)


# ============================================================
# 9. 防止多个请求同时调用generate
# ============================================================

generation_lock = threading.Lock()


# ============================================================
# 10. FastAPI
# ============================================================

app = FastAPI(
    title="QualityPilot PCB Qwen API",
    version="1.0.0",
    description=(
        "Qwen2.5-7B-Instruct + PCB QLoRA "
        "OpenAI-compatible API"
    ),
)


# ============================================================
# 11. OpenAI-like数据结构
# ============================================================

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):

    model: Optional[str] = MODEL_NAME

    messages: List[Message]

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    max_tokens: int = Field(
        default=512,
        ge=1,
        le=2048,
    )

    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
    )

    stream: bool = False


# ============================================================
# 12. API Key验证
# ============================================================

def verify_api_key(
    authorization: Optional[str]
):

    if not API_KEY:
        raise HTTPException(status_code=503, detail="API key is not configured")

    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
        )

    expected = f"Bearer {API_KEY}"

    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )


# ============================================================
# 13. Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",

        "model": MODEL_NAME,

        "base_model": "Qwen2.5-7B-Instruct",

        "adapter": "PCB-QLoRA",

        "gpu": torch.cuda.get_device_name(0),

        "gpu_memory_allocated_gb": round(
            torch.cuda.memory_allocated()
            / 1024**3,
            2,
        ),
    }


# ============================================================
# 14. OpenAI /v1/models
# ============================================================

@app.get("/v1/models")
def models(
    authorization: Optional[str] = Header(
        default=None
    )
):

    verify_api_key(
        authorization
    )

    return {
        "object": "list",

        "data": [
            {
                "id": MODEL_NAME,

                "object": "model",

                "created": int(
                    time.time()
                ),

                "owned_by": "qualitypilot",
            }
        ],
    }


# ============================================================
# 15. 构造Message
# ============================================================

def build_messages(
    request_messages: List[Message]
):

    messages = [
        {
            "role": "system",
            "content": GROUNDING_SYSTEM_PROMPT,
        }
    ]

    # 保留调用者自己传来的消息
    for msg in request_messages:

        # 避免重复插入我们自己的system
        messages.append(
            {
                "role": msg.role,
                "content": msg.content,
            }
        )

    return messages


# ============================================================
# 16. Chat Completions
# ============================================================

@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionRequest,

    authorization: Optional[str] = Header(
        default=None
    ),
):

    verify_api_key(
        authorization
    )

    # 第一版暂不实现SSE Streaming
    if request.stream:

        raise HTTPException(
            status_code=400,
            detail=(
                "Current server does not support "
                "stream=true yet."
            ),
        )

    if len(request.messages) == 0:

        raise HTTPException(
            status_code=400,
            detail="messages cannot be empty",
        )


    # ========================================================
    # Messages
    # ========================================================

    messages = build_messages(
        request.messages
    )


    # ========================================================
    # Qwen Chat Template
    # ========================================================

    prompt = tokenizer.apply_chat_template(
        messages,

        tokenize=False,

        add_generation_prompt=True,
    )


    # ========================================================
    # Tokenize
    # ========================================================

    inputs = tokenizer(
        prompt,

        return_tensors="pt",
    )

    # 单卡4090D
    inputs = {
        key: value.to("cuda:0")
        for key, value in inputs.items()
    }


    prompt_tokens = (
        inputs["input_ids"].shape[1]
    )


    # ========================================================
    # Generate参数
    # ========================================================

    do_sample = (
        request.temperature > 0
    )

    generate_kwargs = {

        "max_new_tokens":
            request.max_tokens,

        "do_sample":
            do_sample,

        "repetition_penalty":
            1.05,

        "pad_token_id":
            tokenizer.pad_token_id,

        "eos_token_id":
            tokenizer.eos_token_id,
    }


    if do_sample:

        generate_kwargs[
            "temperature"
        ] = request.temperature

        generate_kwargs[
            "top_p"
        ] = request.top_p


    # ========================================================
    # Generation
    # ========================================================

    start_time = time.time()

    with generation_lock:

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                **generate_kwargs,
            )


    latency = (
        time.time()
        - start_time
    )


    # ========================================================
    # 只取新生成内容
    # ========================================================

    generated_ids = outputs[
        0,
        prompt_tokens:
    ]


    answer = tokenizer.decode(
        generated_ids,

        skip_special_tokens=True,
    ).strip()


    completion_tokens = len(
        generated_ids
    )


    # ========================================================
    # OpenAI-compatible response
    # ========================================================

    response = {

        "id": (
            "chatcmpl-"
            + uuid.uuid4().hex
        ),

        "object":
            "chat.completion",

        "created":
            int(time.time()),

        "model":
            MODEL_NAME,

        "choices": [
            {
                "index": 0,

                "message": {
                    "role": "assistant",
                    "content": answer,
                },

                "finish_reason": "stop",
            }
        ],

        "usage": {
            "prompt_tokens":
                prompt_tokens,

            "completion_tokens":
                completion_tokens,

            "total_tokens":
                (
                    prompt_tokens
                    + completion_tokens
                ),
        },

        # 自定义字段，OpenAI客户端一般会忽略
        "performance": {

            "latency_seconds":
                round(latency, 3),

            "tokens_per_second":
                round(
                    completion_tokens
                    / latency,
                    2,
                )
                if latency > 0
                else 0,
        },
    }

    return response


# ============================================================
# 17. Root
# ============================================================

@app.get("/")
def root():

    return {
        "service":
            "QualityPilot PCB Qwen",

        "model":
            MODEL_NAME,

        "endpoints": [
            "/health",
            "/v1/models",
            "/v1/chat/completions",
        ],
    }