# MLLM-AD：工业缺陷多模态后训练项目（求职导向）

> 目标：10 天内完成“可复现 + 可讲述 + 可演示”的工业多模态后训练项目。

## 当前进度（第 2 轮）
- ✅ 已完成：仓库骨架、路线决策文档、服务器运行指南。
- ✅ 已完成：**数据构建 MVP 脚本**（支持读取 MVTec AD / VisA 并生成指令 JSONL）。
- ⏳ 下一步：你在服务器执行数据脚本，然后我带你做 LoRA 训练脚本与评测脚本。

## 1. 项目任务
1. 是否异常：正常/异常 + 简短解释
2. 缺陷类型：输出缺陷类型 + 理由
3. 缺陷位置：九宫格（左上/上中/右上 ...）

## 2. 推荐路线
- 主线：VisA / MVTec AD + 自动问答构造 + Qwen2.5-VL-3B LoRA/QLoRA
- 增强：后续可吸收 Anomaly-OV 等研究路线的评测方式

## 3. 你的数据路径（已按你提供值写入配置）
在 `configs/data_config.yaml` 中默认设置：
- `mvtec_root: /home/ljh/mvtecAD`
- `visa_root: /home/ljh/VisA`

另外已增加 `allowed_categories`，会按 **MVTec/VisA 官方类别名白名单** 过滤，自动忽略你数据目录里的多余文件夹。

## 4. 快速开始（需要用户在本地/服务器执行）
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.1 先跑数据构建
```bash
python scripts/prepare_data.py --config configs/data_config.yaml
```

预期输出：
- 终端打印 `"[prepare_data] done"`
- 生成文件：
  - `data/processed/train.jsonl`
  - `data/processed/val.jsonl`
  - `data/processed/stats.json`

### 4.2 跑一次健壮性检查
```bash
python scripts/sanity_check.py --jsonl data/processed/train.jsonl
```

预期输出：
- `missing_required_keys=0`
- `missing_images=0`（若不为 0，说明路径或软链接有问题）

## 5. 常见问题
1. `ModuleNotFoundError: src`  
   解决：在仓库根目录运行命令，或执行 `export PYTHONPATH=$(pwd)`。
2. JSONL 为空  
   解决：检查 `configs/data_config.yaml` 里的路径是否真实存在。
3. VisA 结构和脚本假设不一致  
   解决：先跑 `stats.json` 看采样情况，再告诉我你的目录结构截图，我帮你改扫描规则。
4. 目录里有很多杂项文件夹  
   解决：修改 `configs/data_config.yaml` 的 `allowed_categories.mvtec / allowed_categories.visa`，只保留你要训练的类别。

## 6. 下一步计划
- [已提供] 训练脚本（`scripts/train_lora.py`）+ 训练配置（`configs/train_lora.yaml`）
- [下一轮] 评测脚本（`scripts/evaluate.py`）+ baseline 对比
- [后续] Gradio Demo

## 7. 下一步怎么训练（需要用户在本地/服务器执行）
```bash
python scripts/train_lora.py --config configs/train_lora.yaml
```

建议先用 `configs/train_lora.yaml` 的小样本设置快速验证（`max_train_samples=3000`）。

训练成功的典型标志：
- 终端会先打印 LoRA 可训练参数比例；
- 日志中出现 `loss` 持续输出；
- `outputs/checkpoints/qwen25vl_lora/` 下出现 adapter 权重与 processor 文件。

### 若出现 `torch<2.4 + transformers>=5` 报错（你的环境就是这个情况）
不需要升级 PyTorch，只要把 transformers 固定到 4.x：

```bash
pip install --upgrade --no-deps "transformers==4.47.1" "tokenizers==0.21.0"
```
