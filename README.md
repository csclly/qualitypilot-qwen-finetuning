# QualityPilot Qwen2.5 Fine-tuning

PCB / AOI 场景的 Qwen2.5-7B-Instruct QLoRA 训练与对比推理项目，来自用户已运行的云端训练脚本。应用仓库：[QualityPilot](https://github.com/csclly/qualitypilot)。

本仓库保存代码、观察到的环境版本、适配器配置与训练记录，不包含基础模型、LoRA 权重或私有训练数据。它是训练实验的可审阅代码快照；尚未在全新环境完整重训，不能将源码整理视为训练复现完成。

## 内容

| 文件 | 用途 |
| --- | --- |
| train.py | NF4 4-bit QLoRA，训练最终 assistant 回复，过滤没有监督 token 的样本 |
| inference.py | 单个问题分别运行基础模型和 LoRA |
| compare_eval.py | 10 题基础模型/LoRA 原文对比，记录生成速度与关键词覆盖 |
| test_load.py / test_lora.py | GPU 环境手动加载检查；不会被 pytest 默认收集 |
| model_server.py | 已有模型的 FastAPI 非流式聊天服务与健康检查 |
| inspect_dataset.py | 检查 messages JSONL 格式、精确重复与跨集合相同提示，不导出文本 |
| metadata/ | 实际环境版本、原始脚本摘要、适配器配置和训练状态 |
| tests/ | 无需 GPU 的预处理与数据检查测试 |
| docs/ | 训练记录说明、应用侧评测边界 |

FastAPI 服务源码已收录，启动方法见下文。原训练数据的生成、清洗、划分脚本尚未收到，不能宣称完整数据流水线已经复现。

## 已核实的实验配置与结果

来源为上传的 train.py、adapter_config.json 和 trainer_state.json，不是本轮重新训练：

| 项目 | 观察值 |
| --- | --- |
| 基础模型 | Qwen2.5-7B-Instruct；精确上游 revision 未记录 |
| 量化 | NF4 4-bit、double quant、BF16 计算 |
| LoRA | r=16、alpha=32、dropout=0.05 |
| 目标模块 | q_proj、k_proj、v_proj、o_proj |
| 最大长度 | 1024 tokens |
| 单设备 batch / 梯度累积 | 1 / 8（单设备有效 batch 8） |
| 训练轮数 / 步数 | 2 / 1250 |
| 学习率 / 调度 | 2e-4 / cosine，warmup 40 steps |
| 随机种子 | 42 |
| 最终验证损失 | 0.09870848059654236 |
| 整体训练平均损失 | 0.14754383438825608 |
| 训练记录中的 runtime | 3369.9955 秒，约 56.2 分钟 |

源文件使用 train_5000 / test_500 的文件名，但实际数据内容和有效样本数尚未独立检查。test_500 在训练时用于周期性验证，所以不能再将其作为完全未参与开发的最终测试集。损失值不等于业务正确率，也不证明没有数据泄漏。

## 环境与准备

用户提供的实际环境为 PyTorch 2.6.0+cu124、Transformers 5.16.1、PEFT 0.20.0 等，完整记录见 metadata/environment.observed.json。Python 与驱动版本尚未采集。

优先在原 qwen-sft 环境中复现。新环境先安装与 CUDA 匹配的 PyTorch，再安装 requirements.txt；文件固定了直接依赖版本，不是完整环境锁文件。此次没有重新安装全部 GPU 依赖。

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

从仓库根目录运行。将本地路径通过环境变量传入：

```bash
export PCB_BASE_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
export PCB_TRAIN_FILE=/root/autodl-tmp/pcb-qwen-sft/data/pcb_qwen_sft_train_5000.jsonl
export PCB_EVAL_FILE=/root/autodl-tmp/pcb-qwen-sft/data/pcb_qwen_sft_test_500.jsonl
export PCB_OUTPUT_DIR=/root/autodl-tmp/pcb-qwen-sft/output/new-experiment
```

不要将 PCB_OUTPUT_DIR 指向已有训练成果；训练脚本拒绝非空输出目录，保留既有权重。原脚本的样本文本调试输出默认关闭，仅明确设置 PCB_DEBUG_SAMPLES=1 时输出，不应将含私有样本的日志上传。

先检查数据：

```bash
python inspect_dataset.py --train "$PCB_TRAIN_FILE" --validation "$PCB_EVAL_FILE" --output eval_results/data-check.json
```

实际训练会占用 GPU、写入新检查点并持续一段时间，确认当前 GPU 资源安排后执行：

```bash
python train.py
```

本轮整理不自动启动训练，不修改当前云端在线模型。

## 运行已有 adapter

```bash
export PCB_LORA_PATH=/root/autodl-tmp/pcb-qwen-sft/output/pcb-lora-5000-assistant-only
python inference.py
export PCB_COMPARISON_OUTPUT_DIR=eval_results/comparison-run-01
python compare_eval.py
```

compare_eval.py 先对全部题运行基础模型，再在同一基础模型加载 LoRA。原文与统计保存到新目录中的 base_vs_lora.jsonl。已有文件会被拒绝覆盖。它是自由回答关键词覆盖实验，不测 QualityPilot 五字段 JSON，也不自动判断事实忠实度。当前未收到原 base_vs_lora 结果，不能编造微调提升。

## 启动模型服务

使用已有基础模型和 adapter，在云端 qwen-sft 环境、仓库根目录执行：

```bash
export PCB_BASE_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
export PCB_LORA_PATH=/root/autodl-tmp/pcb-qwen-sft/output/pcb-lora-5000-assistant-only
read -r -s -p "Model API key: " PCB_LLM_API_KEY
export PCB_LLM_API_KEY
python -m uvicorn model_server:app --host 127.0.0.1 --port 8001 --workers 1
```

密钥必须与 QualityPilot 后端 GENERATION_API_KEY 一致。服务拒绝空密钥；不要将真实密钥写入示例文件或提交 Git。单 GPU 使用单 worker，避免重复加载模型。直接执行 python model_server.py 只会加载模型并退出，不会启动 HTTP 服务；等待 Uvicorn 显示启动完成后，在另一个云端终端检查：

```bash
curl --fail http://127.0.0.1:8001/health
```

本地通过 SSH 将 18001 转发到云端 8001，QualityPilot 设置 GENERATION_PROVIDER=openai_compatible，并使用 GENERATION_BASE_URL=http://127.0.0.1:18001/v1、GENERATION_MODEL=pcb-qwen-lora。现有端口转发可继续使用。接口包括 /health、/v1/models、/v1/chat/completions；聊天与模型列表需要 Bearer 认证。仅支持非流式文本聊天，尚无服务端 JSON 约束解码或工具执行。

此代码保留原服务的系统提示与推理逻辑。当前 model 字段不校验别名，finish_reason 固定为 stop，尚未实现输入长度限制、负载限流和准确的截断标记；不应把它描述为完整兼容所有 OpenAI 接口。整理后的路径和认证变更仅经过本地无 GPU 检查，未覆盖当前云端文件或重新部署。

## 测试与数据样例

```bash
python -m pytest -q
python inspect_dataset.py --train data/examples/train.example.jsonl --validation data/examples/validation.example.jsonl --output eval_results/example-data-check.json
```

样例数据是新编写的 3 条合成格式演示，不是原始 5000/500 数据的抽样，也不用于证明模型效果。无 GPU 测试直接提取并执行原训练预处理函数，覆盖最终 assistant 标签、截断过滤、数据格式和重复检测；它不能替代真实 Qwen tokenizer 与 GPU 训练验证。

## 应用侧评测

QualityPilot 的固定证据八题评测已经实际执行：5/8 通过结构与引用契约（62.5%），业务质量尚未打分。低验证损失与应用失败同时存在，后续应检查数据分布、格式一致性和提示模板。详见 [应用评测报告](https://github.com/csclly/qualitypilot/blob/main/backend/evaluation/generation_v1/reports/2026-09-03-analysis.md)。

数据使用授权和模型权重许可应按实际来源确认；当前不重新发布数据或权重。
