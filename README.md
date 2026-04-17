# MLLM-AD：工业缺陷多模态后训练项目（求职导向）

> 目标：在约 10 天内完成一个可复现、可讲述、可演示的工业异常多模态项目（数据构建 + LoRA/QLoRA 微调 + 评测 + Demo + 文档）。

## 1. 项目背景
工业视觉场景中，传统异常检测方法通常只给分数或热图，难以直接“解释为什么异常”。
本项目聚焦 **多模态后训练（post-training）**：基于开源视觉语言模型，在工业数据上做指令化微调，让模型输出：
- 是否异常（normal / anomaly）
- 缺陷类型（如 crack / scratch / contamination）
- 缺陷位置（九宫格粗定位）
- 简短判断依据（可面试讲述）

## 2. 项目目标
- 形成一个完整仓库（代码 + 配置 + 文档 + Demo）。
- 支持最小闭环：数据准备 → 训练 → 评测 → 推理 → Demo。
- 输出可写入简历、可用于面试复盘的项目材料。

## 3. 技术路线（第一轮决策）
- 先评估路线 A（Anomaly-Instruct-125k / Anomaly-OV / VisA-D&R）。
- 若存在落地风险，则降级到路线 B（VisA / MVTec AD + 自动问答构造 + Qwen2.5-VL-3B LoRA）。
- 当前默认推荐路线：**路线 B（稳妥优先）**，路线 A 作为可选增强。

详见：`docs/decision_log.md`。

## 4. 数据来源（计划）
- 首选稳定公开工业数据：**VisA、MVTec AD**。
- 使用数据集原有标签/目录结构/mask 自动构建多任务指令数据（JSONL）。
- 位置任务优先使用 mask 生成九宫格位置；无 mask 时用弱监督规则并标注局限。

## 5. 安装方式（需要用户在本地/服务器执行）
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

预期结果：
- `pip install` 无报错；
- 可执行 `python -c "import torch; print(torch.__version__)"` 输出版本号。

## 6. 数据准备（占位，后续补全）
```bash
# 需要用户在本地/服务器执行（示例）
python scripts/prepare_data.py --config configs/data_config.yaml
```

## 7. 训练（占位，后续补全）
```bash
# 需要用户在本地/服务器执行（示例）
python scripts/train_lora.py --config configs/train_lora.yaml
```

## 8. 评测（占位，后续补全）
```bash
# 需要用户在本地/服务器执行（示例）
python scripts/evaluate.py --config configs/eval_config.yaml
```

## 9. Demo（占位，后续补全）
```bash
# 需要用户在本地/服务器执行（示例）
python demo/app.py
```

## 10. 项目结果展示位
- Baseline vs LoRA 对比表（Accuracy/F1/位置准确率）
- 成功/失败案例可视化
- Demo 截图与典型问答

## 11. 已知局限
- 第一阶段以“可跑通 + 可讲述”为目标，非 SOTA 追求。
- 九宫格位置是粗粒度定位，不等价于精细分割。
- 若缺少 pixel-level mask，位置标签可能存在弱监督噪声。

## 12. 后续扩展方向
- 引入更高质量指令模板和困难负样本。
- 加入多轮追问（root cause / repair suggestion）。
- 扩展到跨域（工业 + 医疗 + 3D）异常推理。
