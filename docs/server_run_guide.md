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
