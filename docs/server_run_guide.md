# server_run_guide：服务器执行指南（第二版）

> 说明：以下命令 **需要你在本地/服务器执行**。我不会伪造训练/评测已完成。

## 0. 你现在该做什么（最短路径）
你已经有数据：
- `/home/ljh/mvtecAD`
- `/home/ljh/VisA`

所以你现在只要做这 4 步：
1) 安装依赖
2) 运行数据构建脚本
3) 做 sanity check
4) 把输出统计贴给我，我继续给你训练与评测脚本

---

## 1. 环境准备
```bash
cd /workspace/MLLM-AD
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

成功标志：
- `pip install` 无报错
- `python -c "import torch, transformers; print(torch.__version__)"` 有版本输出

---

## 2. 确认配置路径
打开 `configs/data_config.yaml`，检查：
```yaml
paths:
  mvtec_root: /home/ljh/mvtecAD
  visa_root: /home/ljh/VisA
```

如果你的真实路径不同，请改这里。

并确认 `allowed_categories`：脚本只会处理白名单类别，自动跳过多余文件夹。

---

## 3. 构建指令数据（MVP）
```bash
python scripts/prepare_data.py --config configs/data_config.yaml
```

成功标志：
- 终端出现 `[prepare_data] done`
- 生成以下文件：
  - `data/processed/train.jsonl`
  - `data/processed/val.jsonl`
  - `data/processed/stats.json`

---

## 4. 运行数据健壮性检查
```bash
python scripts/sanity_check.py --jsonl data/processed/train.jsonl
```

成功标志：
- `missing_required_keys=0`
- `missing_images=0`

---

## 5. 你执行完后发我这三段输出
请把下面命令的输出发给我：
```bash
cat data/processed/stats.json
python scripts/sanity_check.py --jsonl data/processed/train.jsonl
python scripts/sanity_check.py --jsonl data/processed/val.jsonl
```

我会基于你的真实统计结果，给你下一轮：
- 训练配置（quick + full）
- LoRA/QLoRA 超参数建议
- 评测脚本与 baseline 对比流程

---

## 6. LoRA/QLoRA 训练（已可直接运行）
```bash
python scripts/train_lora.py --config configs/train_lora.yaml
```

当前默认是 LoRA（`use_qlora: false`），先保证稳定跑通。  
建议先不改配置直接跑 quick 版本，确认链路通，再把 `max_train_samples` 设为 `null` 跑全量。

成功标志：
- 终端打印 LoRA trainable 参数统计；
- 日志持续输出 `loss`；
- `outputs/checkpoints/qwen25vl_lora/` 出现保存文件。

---

## 常见报错与排查
1. `FileNotFoundError`（找不到数据路径）
- 检查 `configs/data_config.yaml`
- 检查挂载路径权限

2. `ModuleNotFoundError: src`
- 在仓库根目录执行命令
- 或执行 `export PYTHONPATH=$(pwd)`

3. `missing_images > 0`
- 检查 JSONL 中 `image` 字段是否为绝对路径
- 检查是否有失效软链接
4. `CUDA out of memory`
- 降低 `per_device_train_batch_size`
- 提高 `gradient_accumulation_steps`
- 降低 `max_seq_length`
- 保持 `use_qlora: true`
5. `Disabling PyTorch because PyTorch >= 2.4 is required but found 2.1.2`
- 这是 `transformers>=5` 与 `torch==2.1.x` 不兼容导致
- 不要升级 torch，执行：
  ```bash
  pip install --upgrade --no-deps "transformers==4.50.3" "tokenizers==0.21.1"
  ```
- 然后重新运行：
  ```bash
  python scripts/train_lora.py --config configs/train_lora.yaml
  ```
6. `huggingface-hub>=0.24,<1.0 is required ... but found 1.11.0`
- 这是 `transformers 4.x` 和 `huggingface-hub 1.x` 冲突
- 执行：
  ```bash
  pip install --upgrade --no-deps "huggingface-hub==0.36.0"
  ```
- 然后重新运行训练命令
7. `cannot import name '_safe_open_handle' from safetensors._safetensors_rust`
- 这是 `safetensors` 二进制扩展与当前环境不匹配
- 执行：
  ```bash
  pip install --force-reinstall --no-cache-dir --no-deps "safetensors==0.4.5"
  ```
- 再执行训练命令
8. `PackageNotFoundError: No package metadata was found for bitsandbytes`
- 说明环境里没有 bitsandbytes
- 当前脚本默认会自动降级到普通 LoRA 继续跑
- 若要坚持 QLoRA，请安装：
  ```bash
  pip install bitsandbytes==0.44.1
  ```
- 并把 `configs/train_lora.yaml` 中 `model.use_qlora` 设为 `true`
9. `Connection to huggingface.co timed out` / `We couldn't connect to https://huggingface.co`
- 这是网络不可达，不是代码错误
- 推荐离线方案：
  1) 在可联网机器下载模型到本地目录  
     `huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /home/ljh/models/Qwen2.5-VL-3B-Instruct`
  2) 配置 `configs/train_lora.yaml`：
     - `model.local_model_path: /home/ljh/models/Qwen2.5-VL-3B-Instruct`
     - `model.local_files_only: true`
  3) 重新训练
- 若有镜像，可设置：`model.hf_endpoint: https://hf-mirror.com`
10. `AttributeError: 'NoneType' object has no attribute 'split'`（发生在版本检查）
- 说明某些包版本元信息异常或缺失
- 执行：
  ```bash
  pip install --upgrade --no-deps "transformers==4.50.3" "tokenizers==0.21.1" "huggingface-hub==0.36.0"
  ```
- 再重新运行训练

---

## Git 说明（关于“直接推送到 main”）
当前这个 Codex 运行环境通常不会自动配置你的远程仓库凭据与 `origin`。  
如果你要把改动推到远程 `main`，请在你本地/服务器执行：

```bash
git checkout main
git merge --ff-only work
git push origin main
```

如果 `main` 不存在，可先创建：

```bash
git checkout -b main
git merge --ff-only work
git push -u origin main
```

---

## 服务器已是旧版 main 时，如何“一键更新到最新版”
如果你的服务器目录是直接 `git clone` 的第一版 main，并且你不想处理冲突，建议使用**强制对齐**（会丢弃本地未提交修改）：

```bash
cd /path/to/MLLM-AD
git fetch origin
git checkout main
git reset --hard origin/main
git clean -fd
```

成功标志：
- `git status` 显示 working tree clean
- `git log --oneline -n 1` 显示最新 main 提交

> 如果你服务器上有自己改过但未提交的文件，请先备份目录，再执行上述命令。
