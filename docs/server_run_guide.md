# server_run_guide：服务器执行指南（第一版）

> 说明：以下命令 **需要用户在本地/服务器执行**。本仓库代理不会假装已执行训练/评测。

## 1. 环境准备
```bash
cd /path/to/MLLM-AD
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

成功标志：
- `pip install` 无报错。
- `python -c "import torch, transformers; print(torch.__version__)"` 能输出版本。

常见报错：
- `bitsandbytes` 与 CUDA 版本不兼容。
- `torch` 版本与驱动不匹配。

排查建议：
- 先确认 `nvidia-smi` 可用。
- 固定 CUDA 对应的 torch 版本重新安装。

## 2. 数据准备（占位）
```bash
# 需要用户在本地/服务器执行（后续脚本完善）
python scripts/prepare_data.py --config configs/data_config.yaml
```

成功标志（预期）：
- 在 `data/processed/` 生成 `train.jsonl`、`val.jsonl`。
- 终端输出样本统计与类别分布。

## 3. 训练（占位）
```bash
# 需要用户在本地/服务器执行（后续脚本完善）
python scripts/train_lora.py --config configs/train_lora.yaml
```

成功标志（预期）：
- `outputs/checkpoints/` 出现 checkpoint。
- 日志含有 train loss 下降趋势。

## 4. 评测（占位）
```bash
# 需要用户在本地/服务器执行（后续脚本完善）
python scripts/evaluate.py --config configs/eval_config.yaml
```

成功标志（预期）：
- 输出 Accuracy/F1/位置准确率。
- 导出成功/失败案例到 `outputs/figures/`。

## 5. Demo（占位）
```bash
# 需要用户在本地/服务器执行（后续脚本完善）
python demo/app.py
```

成功标志（预期）：
- 终端显示本地 URL（如 `http://127.0.0.1:7860`）。
- 网页可上传图片并返回四项结果。

## 6. 推荐执行顺序（MVP）
1. 安装依赖
2. 先跑小样本数据准备
3. 先跑小配置训练（快速验证）
4. 跑评测并导出案例
5. 启动 demo 检查端到端流程
