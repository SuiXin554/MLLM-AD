# decision_log：路线与取舍记录（第一版）

## 决策日期
- 2026-04-17

## 候选路线
### 路线 A（论文/项目链）
- Anomaly-Instruct-125k
- Anomaly-OV
- VisA-D&R

### 路线 B（稳妥工程路线）
- 使用 VisA / MVTec AD
- 自动构造图像-指令数据
- 使用 Qwen2.5-VL-3B 做 LoRA/QLoRA

## 对路线 A 的可行性判断（首轮）
### 优点
- 任务定义完整，贴近“异常检测 + 推理”。
- 有公开论文与代码，面试故事性强。

### 风险
1. 数据依赖链较复杂（项目页 + 外部盘 + 额外数据配置）。
2. 训练脚本偏研究原型，改造到 10 天求职项目有不确定性。
3. 复现实验成本较高，容易在环境或数据准备上耗时。

## 结论
- **默认推荐路线 B**：优先保证 10 天可交付与可复现。
- 路线 A 作为“增强选项”：
  - 若第 6-8 天进度充足，再引入 A 的部分数据或评测设定进行加分。

## Trade-off 摘要
- 牺牲：前沿 novelty。
- 换取：稳定落地、可讲述、可验证、可复现。

## 进展更新（2026-04-17 第二轮）
- 用户已具备本地数据：`/home/ljh/mvtecAD` 与 `/home/ljh/VisA`。
- 因此优先推进“数据构建 MVP → 训练脚本 MVP”，不再停留在文档占位。
- 新增 `scripts/prepare_data.py` 与 `scripts/sanity_check.py`，先保证数据闭环真实可执行。

## 进展更新（2026-04-19）
- 用户反馈数据目录中存在多余文件夹，新增 `allowed_categories` 白名单机制。
- 数据构建阶段按 MVTec/VisA 标准类别过滤，避免脏目录污染训练 JSONL。

## 进展更新（2026-04-19 训练兼容性）
- 用户环境为 `torch==2.1.2`，且安装了 `transformers==5.5.4`，二者不兼容会导致训练入口报错。
- 方案：保持 PyTorch 不变，仅将 transformers/tokenizers 固定到 4.x 兼容组合。
- 补充：`transformers 4.x` 需要 `huggingface-hub<1.0`，若为 1.x 需降级到 0.36.0。
- 补充：`safetensors` 若出现 `_safe_open_handle` 导入错误，需强制重装同版本 wheel（0.4.5）。
- 补充：若缺失 `bitsandbytes`，训练脚本默认自动从 QLoRA 降级为 LoRA，避免流程中断。
