import os
import torch
import transformers

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# 0. 基础配置
# ============================================================

SEED = 42

set_seed(SEED)

MODEL_PATH = os.getenv("PCB_BASE_MODEL_PATH", "models/Qwen2.5-7B-Instruct")

TRAIN_FILE = os.getenv("PCB_TRAIN_FILE", "data/private/train.jsonl")

EVAL_FILE = os.getenv("PCB_EVAL_FILE", "data/private/validation.jsonl")

OUTPUT_DIR = os.getenv("PCB_OUTPUT_DIR", "output/pcb-lora-run")

MAX_LENGTH = 1024


# ============================================================
# 1. 打印环境信息
# ============================================================

print("=" * 70)
print("环境信息")
print("=" * 70)

print("Transformers:", transformers.__version__)
print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU显存:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2,
        ),
        "GB",
    )
else:
    raise RuntimeError("没有检测到 CUDA GPU")


# ============================================================
# 2. 路径检查
# ============================================================

print("\n" + "=" * 70)
print("路径检查")
print("=" * 70)

for path in [
    MODEL_PATH,
    TRAIN_FILE,
    EVAL_FILE,
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"文件或目录不存在：{path}"
        )

if os.path.isdir(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
    raise FileExistsError("训练输出目录非空；请设置新的 PCB_OUTPUT_DIR，避免覆盖已有 adapter。")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

print("模型路径：", MODEL_PATH)
print("训练集：", TRAIN_FILE)
print("验证集：", EVAL_FILE)
print("输出目录：", OUTPUT_DIR)


# ============================================================
# 3. Tokenizer
# ============================================================

print("\n" + "=" * 70)
print("加载 Tokenizer")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

# Qwen 通常有 pad token，但为了保险
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Causal LM 训练建议右侧 padding
tokenizer.padding_side = "right"

print("pad_token:", tokenizer.pad_token)
print("pad_token_id:", tokenizer.pad_token_id)

print("eos_token:", tokenizer.eos_token)
print("eos_token_id:", tokenizer.eos_token_id)


# ============================================================
# 4. QLoRA 4-bit 配置
# ============================================================

print("\n" + "=" * 70)
print("配置 QLoRA 4-bit")
print("=" * 70)

bnb_config = BitsAndBytesConfig(

    # 基础模型权重以 4 bit 加载
    load_in_4bit=True,

    # QLoRA 推荐 NF4
    bnb_4bit_quant_type="nf4",

    # 运算使用 BF16
    bnb_4bit_compute_dtype=torch.bfloat16,

    # Double Quantization
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 5. 加载 Qwen2.5-7B
# ============================================================

print("\n" + "=" * 70)
print("加载 Qwen2.5-7B")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,

    quantization_config=bnb_config,

    device_map="auto",

    trust_remote_code=True,
)

# 训练时不能使用 KV cache
model.config.use_cache = False

print("基础模型加载完成")

print(
    "当前显存 allocated:",
    round(
        torch.cuda.memory_allocated()
        / 1024**3,
        2,
    ),
    "GB",
)

print(
    "当前显存 reserved:",
    round(
        torch.cuda.memory_reserved()
        / 1024**3,
        2,
    ),
    "GB",
)


# ============================================================
# 6. 为 QLoRA 做准备
# ============================================================

print("\n" + "=" * 70)
print("准备 k-bit 训练")
print("=" * 70)

model = prepare_model_for_kbit_training(
    model,
)

# 减少 activation 显存
model.gradient_checkpointing_enable()


# ============================================================
# 7. LoRA 配置
# ============================================================

print("\n" + "=" * 70)
print("添加 LoRA Adapter")
print("=" * 70)

lora_config = LoraConfig(

    # LoRA Rank
    r=16,

    # LoRA Scaling
    lora_alpha=32,

    # Dropout
    lora_dropout=0.05,

    # 当前先微调 Attention 的四个投影
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    bias="none",

    task_type="CAUSAL_LM",
)

model = get_peft_model(
    model,
    lora_config,
)

print("\n===== LoRA 参数情况 =====")

model.print_trainable_parameters()


# ============================================================
# 8. 加载训练数据
# ============================================================

print("\n" + "=" * 70)
print("加载训练数据")
print("=" * 70)

train_dataset = load_dataset(
    "json",
    data_files=TRAIN_FILE,
    split="train",
)

eval_dataset = load_dataset(
    "json",
    data_files=EVAL_FILE,
    split="train",
)

print("训练样本数：", len(train_dataset))
print("验证样本数：", len(eval_dataset))

print("\n===== 第一条训练数据 =====")

if os.getenv("PCB_DEBUG_SAMPLES") == "1":
    print(train_dataset[0])


# ============================================================
# 9. Assistant-only Loss 预处理
# ============================================================

def preprocess(example):
    """
    输入格式：

    {
        "messages": [
            {
                "role": "user",
                "content": "..."
            },
            {
                "role": "assistant",
                "content": "..."
            }
        ]
    }

    Assistant-only Loss：

    system
    user
    assistant 起始标记

        ↓

    labels = -100

    assistant 正文

        ↓

    labels = 对应 token id
    """

    messages = example["messages"]

    # --------------------------------------------------------
    # 数据格式检查
    # --------------------------------------------------------

    if len(messages) < 2:
        raise ValueError(
            "messages 至少需要 user + assistant"
        )

    if messages[-1]["role"] != "assistant":
        raise ValueError(
            "当前数据最后一条 message 必须是 assistant"
        )

    # --------------------------------------------------------
    # 只留下 assistant 之前的 messages
    # --------------------------------------------------------

    prompt_messages = messages[:-1]

    # --------------------------------------------------------
    # Prompt:
    #
    # system
    # user
    # assistant start token
    # --------------------------------------------------------

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # --------------------------------------------------------
    # Full conversation:
    #
    # system
    # user
    # assistant
    # answer
    # end
    # --------------------------------------------------------

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # --------------------------------------------------------
    # Tokenize Prompt
    # --------------------------------------------------------

    prompt_tokens = tokenizer(
        prompt_text,

        add_special_tokens=False,

        truncation=True,

        max_length=MAX_LENGTH,
    )

    # --------------------------------------------------------
    # Tokenize Full Conversation
    # --------------------------------------------------------

    full_tokens = tokenizer(
        full_text,

        add_special_tokens=False,

        truncation=True,

        max_length=MAX_LENGTH,
    )

    input_ids = full_tokens["input_ids"]

    attention_mask = full_tokens[
        "attention_mask"
    ]

    # --------------------------------------------------------
    # 构造 labels
    # --------------------------------------------------------

    labels = input_ids.copy()

    prompt_length = len(
        prompt_tokens["input_ids"]
    )

    # --------------------------------------------------------
    # 如果 prompt 已经占满 MAX_LENGTH
    # 那么 assistant 内容完全被截断
    # --------------------------------------------------------

    if prompt_length >= len(labels):

        labels = [
            -100
            for _ in labels
        ]

    else:

        # ----------------------------------------------------
        # System + User + Assistant Start
        # 全部不计算 Loss
        # ----------------------------------------------------

        for i in range(prompt_length):
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ============================================================
# 10. Tokenize Train Dataset
# ============================================================

print("\n" + "=" * 70)
print("Tokenize 训练集")
print("=" * 70)

tokenized_train_dataset = train_dataset.map(
    preprocess,

    remove_columns=train_dataset.column_names,

    desc="Tokenizing train dataset",
)


# ============================================================
# 11. Tokenize Eval Dataset
# ============================================================

print("\n" + "=" * 70)
print("Tokenize 验证集")
print("=" * 70)

tokenized_eval_dataset = eval_dataset.map(
    preprocess,

    remove_columns=eval_dataset.column_names,

    desc="Tokenizing eval dataset",
)


# ============================================================
# 12. 检查 Assistant-only Loss
# ============================================================

print("\n" + "=" * 70)
print("Assistant-only Loss 检查")
print("=" * 70)

sample = tokenized_train_dataset[0]

input_ids = sample["input_ids"]

labels = sample["labels"]

print(
    "input_ids 长度:",
    len(input_ids),
)

print(
    "labels 长度:",
    len(labels),
)

masked_count = sum(
    1
    for x in labels
    if x == -100
)

train_count = sum(
    1
    for x in labels
    if x != -100
)

print(
    "不参与 loss 的 token 数:",
    masked_count,
)

print(
    "参与 loss 的 token 数:",
    train_count,
)

if train_count == 0:
    raise RuntimeError(
        "第一条训练数据没有 assistant token "
        "参与 Loss，请检查数据或 MAX_LENGTH"
    )

# ------------------------------------------------------------
# 找到 assistant 正文第一个 token
# ------------------------------------------------------------

first_train_index = next(
    i
    for i, label in enumerate(labels)
    if label != -100
)

masked_text = tokenizer.decode(
    input_ids[:first_train_index],

    skip_special_tokens=False,
)

assistant_text = tokenizer.decode(
    input_ids[first_train_index:],

    skip_special_tokens=False,
)

print("\n===== 不参与 Loss 的部分 =====")

if os.getenv("PCB_DEBUG_SAMPLES") == "1":
    print(masked_text)

print("\n===== 参与 Loss 的部分 =====")

if os.getenv("PCB_DEBUG_SAMPLES") == "1":
    print(assistant_text)


# ============================================================
# 13. 过滤无有效 Assistant token 的数据
# ============================================================

def has_trainable_token(example):

    return any(
        label != -100
        for label in example["labels"]
    )


before_train = len(
    tokenized_train_dataset
)

before_eval = len(
    tokenized_eval_dataset
)


tokenized_train_dataset = (
    tokenized_train_dataset.filter(
        has_trainable_token,
        desc="Filtering invalid train examples",
    )
)

tokenized_eval_dataset = (
    tokenized_eval_dataset.filter(
        has_trainable_token,
        desc="Filtering invalid eval examples",
    )
)


print("\n===== 数据过滤结果 =====")

print(
    "训练集:",
    before_train,
    "->",
    len(tokenized_train_dataset),
)

print(
    "验证集:",
    before_eval,
    "->",
    len(tokenized_eval_dataset),
)


# ============================================================
# 14. Data Collator
# ============================================================

print("\n" + "=" * 70)
print("配置 Data Collator")
print("=" * 70)

data_collator = DataCollatorForSeq2Seq(

    tokenizer=tokenizer,

    model=model,

    # 每个 batch 动态 padding
    padding=True,

    # padding 的 label 不参与 Loss
    label_pad_token_id=-100,

    return_tensors="pt",
)


# ============================================================
# 15. TrainingArguments
#
# 完全按照你当前 transformers 5.16.1
# TrainingArguments.__init__ 的真实签名填写
# ============================================================

print("\n" + "=" * 70)
print("配置 TrainingArguments")
print("=" * 70)

training_args = TrainingArguments(

    # ========================================================
    # 输出路径
    # ========================================================

    output_dir=OUTPUT_DIR,


    # ========================================================
    # 训练轮数
    # ========================================================

    num_train_epochs=2.0,


    # ========================================================
    # Batch
    # ========================================================

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    # 1 × 8
    # effective batch size = 8
    gradient_accumulation_steps=8,


    # ========================================================
    # Learning Rate
    # ========================================================

    learning_rate=2e-4,


    # ========================================================
    # Scheduler
    # ========================================================

    lr_scheduler_type="cosine",

    # 原 warmup_ratio = 0.03
    #
    # 5000 × 2 / 8 ≈ 1250 optimizer steps
    #
    # 1250 × 0.03 ≈ 37.5
    #
    # 所以设置 40 steps
    warmup_steps=40,


    # ========================================================
    # Optimizer
    # ========================================================

    optim="adamw_torch",


    # ========================================================
    # Gradient
    # ========================================================

    max_grad_norm=1.0,


    # ========================================================
    # Precision
    # ========================================================

    bf16=True,

    fp16=False,


    # ========================================================
    # Logging
    # ========================================================

    logging_strategy="steps",

    logging_steps=10,

    logging_first_step=True,


    # ========================================================
    # Evaluation
    # ========================================================

    eval_strategy="steps",

    eval_steps=100,

    # eval 时只需要 loss
    prediction_loss_only=True,


    # ========================================================
    # Saving
    # ========================================================

    save_strategy="steps",

    save_steps=100,

    save_total_limit=2,


    # ========================================================
    # DataLoader
    # ========================================================

    dataloader_num_workers=2,

    dataloader_pin_memory=True,


    # ========================================================
    # Dataset
    # ========================================================

    remove_unused_columns=False,


    # ========================================================
    # 日志平台
    # ========================================================

    report_to="none",


    # ========================================================
    # 随机种子
    # ========================================================

    seed=SEED,

    data_seed=SEED,


    # ========================================================
    # Best Model
    #
    # 第一轮实验先不自动加载最佳模型
    # ========================================================

    load_best_model_at_end=False,
)


# ============================================================
# 16. 打印训练参数摘要
# ============================================================

print("\n===== 训练参数摘要 =====")

print(
    "Epoch:",
    training_args.num_train_epochs,
)

print(
    "Train batch size:",
    training_args.per_device_train_batch_size,
)

print(
    "Gradient accumulation:",
    training_args.gradient_accumulation_steps,
)

effective_batch_size = (
    training_args.per_device_train_batch_size
    *
    training_args.gradient_accumulation_steps
)

print(
    "Effective batch size:",
    effective_batch_size,
)

print(
    "Learning rate:",
    training_args.learning_rate,
)

print(
    "Warmup steps:",
    training_args.warmup_steps,
)

print(
    "LR scheduler:",
    training_args.lr_scheduler_type,
)

print(
    "Eval every:",
    training_args.eval_steps,
    "optimizer steps",
)

print(
    "Save every:",
    training_args.save_steps,
    "optimizer steps",
)


# ============================================================
# 17. 创建 Trainer
# ============================================================

print("\n" + "=" * 70)
print("创建 Trainer")
print("=" * 70)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train_dataset,

    eval_dataset=tokenized_eval_dataset,

    data_collator=data_collator,
)


# ============================================================
# 18. 开始训练
# ============================================================

print("\n")
print("=" * 70)
print("Qwen2.5-7B QLoRA SFT")
print("Assistant-only Loss")
print("=" * 70)

print(
    "训练集样本数：",
    len(tokenized_train_dataset),
)

print(
    "验证集样本数：",
    len(tokenized_eval_dataset),
)

print(
    "Epoch：",
    training_args.num_train_epochs,
)

print(
    "Effective Batch Size：",
    effective_batch_size,
)

estimated_steps = int(
    (
        len(tokenized_train_dataset)
        *
        training_args.num_train_epochs
    )
    /
    effective_batch_size
)

print(
    "预计 Optimizer Steps：约",
    estimated_steps,
)

print(
    "训练前显存 allocated:",
    round(
        torch.cuda.memory_allocated()
        / 1024**3,
        2,
    ),
    "GB",
)

print(
    "训练前显存 reserved:",
    round(
        torch.cuda.memory_reserved()
        / 1024**3,
        2,
    ),
    "GB",
)


# ============================================================
# 19. Train
# ============================================================

train_result = trainer.train()


# ============================================================
# 20. 最终验证
# ============================================================

print("\n")
print("=" * 70)
print("最终 Evaluation")
print("=" * 70)

eval_result = trainer.evaluate()

print("\n===== Eval Result =====")

for key, value in eval_result.items():
    print(
        f"{key}: {value}"
    )


# ============================================================
# 21. 保存最终 LoRA Adapter
# ============================================================

print("\n")
print("=" * 70)
print("保存最终 LoRA Adapter")
print("=" * 70)

model.save_pretrained(
    OUTPUT_DIR,
)

tokenizer.save_pretrained(
    OUTPUT_DIR,
)

trainer.save_state()


print("\nLoRA Adapter 已保存至：")

print(OUTPUT_DIR)


# ============================================================
# 22. 输出最终训练信息
# ============================================================

print("\n")
print("=" * 70)
print("最终 Training Result")
print("=" * 70)

for key, value in train_result.metrics.items():

    print(
        f"{key}: {value}"
    )


print(
    "\n训练结束显存 allocated:",
    round(
        torch.cuda.memory_allocated()
        / 1024**3,
        2,
    ),
    "GB",
)

print(
    "训练结束显存 reserved:",
    round(
        torch.cuda.memory_reserved()
        / 1024**3,
        2,
    ),
    "GB",
)


print("\n")
print("=" * 70)
print("训练全部完成")
print("=" * 70)