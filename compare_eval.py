import os
import json
import time
import torch
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


MODEL_PATH = os.getenv("PCB_BASE_MODEL_PATH", "models/Qwen2.5-7B-Instruct")
LORA_PATH = os.getenv("PCB_LORA_PATH", "output/pcb-lora-run")

OUTPUT_DIR = Path(os.getenv("PCB_COMPARISON_OUTPUT_DIR", "eval_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = OUTPUT_DIR / "base_vs_lora.jsonl"
if RESULT_FILE.exists():
    raise FileExistsError("评测报告已存在；请设置新的 PCB_COMPARISON_OUTPUT_DIR。")


# ============================================================
# 1. 测试问题
# ============================================================

TEST_CASES = [
    {
        "id": "pcb_open_001",
        "category": "PCB根因分析",
        "question": "PCB出现开路缺陷，常见原因有哪些？请给出排查顺序。",
        "keywords": [
            "蚀刻",
            "Gerber",
            "AOI",
            "电测",
            "工艺"
        ],
    },
    {
        "id": "pcb_short_001",
        "category": "PCB根因分析",
        "question": "AOI检测发现某批PCB短路报点突然增加，应该如何分析可能原因？",
        "keywords": [
            "残铜",
            "蚀刻",
            "Gerber",
            "MES",
            "复核"
        ],
    },
    {
        "id": "aoi_false_001",
        "category": "AOI误报分析",
        "question": "AOI检测大量疑似缺口，但人工抽检发现多数为良品，应该如何判断是真缺陷还是误报？",
        "keywords": [
            "原图",
            "Gerber",
            "对位",
            "光照",
            "阈值"
        ],
    },
    {
        "id": "align_001",
        "category": "Gerber与对位",
        "question": "PCB缺陷报点集中在板边，并且不同板的偏移方向基本一致，应该优先排查哪些问题？",
        "keywords": [
            "对位",
            "偏移",
            "Gerber",
            "基准",
            "补偿"
        ],
    },
    {
        "id": "mes_qms_001",
        "category": "MES/QMS",
        "question": "为什么分析PCB质量异常时需要同时查询MES和QMS数据？",
        "keywords": [
            "批次",
            "设备",
            "工艺",
            "缺陷",
            "历史"
        ],
    },
    {
        "id": "agent_plan_001",
        "category": "Agent任务分解",
        "question": (
            "如果让工业智能体自动分析PCB阻焊偏移异常，"
            "请将整个任务拆解成可执行步骤。"
        ),
        "keywords": [
            "事件",
            "MES",
            "Gerber",
            "检索",
            "证据",
            "审批"
        ],
    },
    {
        "id": "agent_tool_001",
        "category": "Agent工具调用",
        "question": (
            "你只能调用MES批次查询、AOI原图、Gerber查询、"
            "QMS历史案例四个工具。分析PCB短路异常时应该按什么顺序调用？"
        ),
        "keywords": [
            "MES",
            "AOI",
            "Gerber",
            "QMS",
            "证据"
        ],
    },
    {
        "id": "uncertain_001",
        "category": "拒绝臆测",
        "question": (
            "AOI发现短路率突然升高，现在只有一张缺陷截图，"
            "没有MES、Gerber和电测数据。请直接告诉我唯一根因。"
        ),
        "keywords": [
            "不能",
            "不足",
            "证据",
            "需要",
            "无法"
        ],
    },
    {
        "id": "risk_001",
        "category": "风险判断",
        "question": (
            "PCB孔铜异常报点升高，什么情况下应该隔离批次，"
            "什么情况下应该先怀疑AOI误报？"
        ),
        "keywords": [
            "电测",
            "复核",
            "批次",
            "真实",
            "误报"
        ],
    },
    {
        "id": "hitl_001",
        "category": "HITL",
        "question": (
            "工业Agent分析出疑似蚀刻异常后，是否应该直接自动停线？"
            "应该怎么设计人工审批？"
        ),
        "keywords": [
            "人工",
            "审批",
            "高风险",
            "停线",
            "证据"
        ],
    },
]


# ============================================================
# 2. QLoRA 4-bit 配置
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 3. Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4. 生成函数
# ============================================================

def generate(model, question):
    messages = [
        {
            "role": "user",
            "content": question,
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
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()

    latency = time.time() - start

    input_length = inputs["input_ids"].shape[1]

    generated_ids = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    output_tokens = len(generated_ids)

    tokens_per_second = (
        output_tokens / latency
        if latency > 0
        else 0
    )

    return {
        "answer": answer,
        "latency": latency,
        "output_tokens": output_tokens,
        "tokens_per_second": tokens_per_second,
    }


# ============================================================
# 5. 简单规则评分
# ============================================================

def keyword_score(answer, keywords):
    hit = []

    for keyword in keywords:
        if keyword.lower() in answer.lower():
            hit.append(keyword)

    return {
        "keyword_hits": hit,
        "keyword_hit_count": len(hit),
        "keyword_total": len(keywords),
        "keyword_score": round(
            len(hit) / len(keywords),
            4,
        ),
    }


def uncertainty_score(answer):
    uncertainty_words = [
        "证据不足",
        "无法确定",
        "不能直接",
        "无法直接",
        "需要进一步",
        "需要补充",
        "不确定",
        "不能仅凭",
    ]

    hits = [
        w
        for w in uncertainty_words
        if w in answer
    ]

    return {
        "uncertainty_hits": hits,
        "uncertainty_score": 1 if hits else 0,
    }


# ============================================================
# 6. 加载 Base
# ============================================================

print("=" * 70)
print("加载 Base Qwen")
print("=" * 70)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

base_model.eval()


# ============================================================
# 7. 先测试 Base
# ============================================================

base_results = {}

for case in TEST_CASES:
    print(
        f"\n[Base] {case['id']} "
        f"{case['category']}"
    )

    result = generate(
        base_model,
        case["question"],
    )

    score = keyword_score(
        result["answer"],
        case["keywords"],
    )

    if case["category"] == "拒绝臆测":
        score.update(
            uncertainty_score(
                result["answer"]
            )
        )

    base_results[case["id"]] = {
        **result,
        **score,
    }

    print(result["answer"][:300])
    print(
        "keyword_score:",
        score["keyword_score"],
    )


# ============================================================
# 8. 加载 LoRA 到 Base
# ============================================================

print("\n" + "=" * 70)
print("加载 LoRA Adapter")
print("=" * 70)

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
)

lora_model.eval()


# ============================================================
# 9. 测试 LoRA
# ============================================================

records = []

for case in TEST_CASES:
    print(
        f"\n[LoRA] {case['id']} "
        f"{case['category']}"
    )

    lora_result = generate(
        lora_model,
        case["question"],
    )

    lora_score = keyword_score(
        lora_result["answer"],
        case["keywords"],
    )

    if case["category"] == "拒绝臆测":
        lora_score.update(
            uncertainty_score(
                lora_result["answer"]
            )
        )

    base_result = base_results[
        case["id"]
    ]

    record = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "keywords": case["keywords"],

        "base_answer": base_result["answer"],
        "base_keyword_score": base_result[
            "keyword_score"
        ],
        "base_latency": round(
            base_result["latency"],
            4,
        ),
        "base_tokens_per_second": round(
            base_result["tokens_per_second"],
            2,
        ),

        "lora_answer": lora_result["answer"],
        "lora_keyword_score": lora_score[
            "keyword_score"
        ],
        "lora_latency": round(
            lora_result["latency"],
            4,
        ),
        "lora_tokens_per_second": round(
            lora_result["tokens_per_second"],
            2,
        ),
    }

    if case["category"] == "拒绝臆测":
        record[
            "base_uncertainty_score"
        ] = base_result.get(
            "uncertainty_score",
            0,
        )

        record[
            "lora_uncertainty_score"
        ] = lora_score.get(
            "uncertainty_score",
            0,
        )

    records.append(record)

    print(lora_result["answer"][:300])
    print(
        "keyword_score:",
        lora_score["keyword_score"],
    )


# ============================================================
# 10. 保存 JSONL
# ============================================================

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8",
) as f:
    for record in records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# 11. 汇总结果
# ============================================================

base_avg = sum(
    x["base_keyword_score"]
    for x in records
) / len(records)

lora_avg = sum(
    x["lora_keyword_score"]
    for x in records
) / len(records)

base_speed = sum(
    x["base_tokens_per_second"]
    for x in records
) / len(records)

lora_speed = sum(
    x["lora_tokens_per_second"]
    for x in records
) / len(records)

print("\n")
print("=" * 70)
print("最终结果")
print("=" * 70)

print(
    "Base 平均关键词覆盖:",
    round(base_avg, 4),
)

print(
    "LoRA 平均关键词覆盖:",
    round(lora_avg, 4),
)

print(
    "Base 平均 tokens/s:",
    round(base_speed, 2),
)

print(
    "LoRA 平均 tokens/s:",
    round(lora_speed, 2),
)

print(
    "\n结果已保存：",
    RESULT_FILE,
)