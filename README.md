# Cognitive Trajectory Prediction

一个可在 CPU 上运行的**合成数据研究原型**：将每次访视的四组数值特征组织为关系图，再用时间感知 LSTM 编码不规则访视历史，预测指定未来日期的五个分位数。项目组合并独立实现已有方法，不主张发明 R-GCN、T-LSTM 或分位数回归。

**English abstract.** This runnable synthetic research prototype combines a typed per-visit graph, a time-aware LSTM, and a baseline-relative ordered quantile head for horizon-conditioned trajectory prediction. It demonstrates a reproducible training and inference workflow; it provides no evidence of clinical performance.

项目地址：[NaotoShirogane331/cognitive-trajectory-prediction](https://github.com/NaotoShirogane331/cognitive-trajectory-prediction) · 版本：0.1.0 · 日期：2026-10-29 · MIT

## 快速开始

需要 Python **3.12**。在项目根目录执行：

```bash
uv sync --extra dev
uv run ctp demo --output outputs/demo
uv run pytest
uv run ruff check .
```

`demo` 实际生成数据、预处理、训练、对测试集预测、评估并绘图，不是预置结果展示。默认随机种子为 `2026`，生成 120 人、每人 6–10 次不规则历史访视，预测最后一次访视之后 90、180、365 天。人员按 70%/15%/15% 划分训练、验证、测试集；训练最多 30 轮，验证 pinball loss 连续 5 轮未改善时提前停止，使用最佳验证检查点。输出目录须为空或不存在；再次运行请换一个新目录。

也可以使用独立虚拟环境和 pip：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ctp demo --output outputs/demo-pip
pytest
ruff check .
```

## 七个命令

以下是完整的分步流程；每个输出路径均为新的目录或文件。参数说明可用 `uv run ctp <command> --help` 查看。

```bash
# 1. 生成合成表和来源说明
uv run ctp generate --output outputs/raw --n-people 120 --seed 2026

# 2. 规范化表、校验关联，生成准备报告
uv run ctp prepare --input outputs/raw --output outputs/prepared

# 3. 检查表结构、时间与索引；报告媒体文件是否存在
uv run ctp check --data outputs/prepared

# 4. 实际 CPU 训练，按验证损失选择检查点
uv run ctp train --data outputs/prepared --output outputs/model \
  --epochs 30 --patience 5 --batch-size 16 --learning-rate 0.001 --seed 2026

# 5. 仅对保存的测试人员预测；省略 --split 时预测所有人员
uv run ctp predict --data outputs/prepared --model outputs/model \
  --output outputs/test-predictions.csv --split test

# 6. 将预测与标签一对一关联，输出指标
uv run ctp evaluate --data outputs/prepared \
  --predictions outputs/test-predictions.csv --output outputs/test-metrics.json

# 7. 一次执行以上流程并绘图
uv run ctp demo --output outputs/demo-full --n-people 120 --epochs 30 --seed 2026
```

模型目录包含 `model.pt`、`preprocessor.json`、`config.json`、`splits.json`、`training_history.csv` 和 `training_summary.json`。预测 CSV 包含人员与日期标识、`horizon_days`、`baseline`、`q05/q25/q50/q75/q95`。评估给出中位数 MAE、pinball loss、名义 90% 区间的经验覆盖率与宽度，以及末次可用分数基线 MAE；这些只是当前合成数据上的指标，不代表真实人群表现。

`demo` 输出目录中的 `raw/` 保存生成表与 `provenance.json`，`prepared/` 保存规范化表与复制的来源说明，`model/` 保存训练状态；`predictions.csv` 仅包含测试集预测，`metrics.json` 保存指标，`trajectory.png` 展示实际预测及起点已可用的历史分数，`summary.json` 汇总流程。图中标记合成数据来源。

## 数据与方法

输入目录包含 `visits.csv`、`assets.csv`、`targets.csv`。每个预测样本是一位人员在一个预测起点对应的一个目标日期；历史包含更早日期的访视及起点访视本身，同日起点的其他会话排除，尚未发布的分数单独屏蔽。缺失值使用独立掩码，零是有效数值。归一化仅拟合训练人员在各自预测起点已知的历史特征。详见[数据格式](docs/data-format.md)。

每次访视包含共享中心节点及 action、assessment、background、condition 四个特征节点；四种关系分别连接中心与对应组。两层 R-GCN 生成中心表示，时间感知 LSTM 使用相邻访视的 `dt_days`，预测头另用 `horizon_days` 控制未来跨度。输出五个严格有序分位数，以最新可用原始分数为基线。详见[方法与公式](docs/method.md)。

## 适用范围与限制

- 示例的 0–30 分是**虚构尺度，不是 MMSE 或 MoCA**；特征名没有预设临床含义。
- 只发布源代码、合成数据与文档，不包含真实人员数据，也不报告临床效果或 PPT 中的提升数字。
- `assets.csv` 仅索引媒体路径。本版本不读取、解码或提取音频/视频特征；合成路径是虚构元数据，不会生成假媒体。
- 五个有序分位数不自动证明概率校准。`q05` 到 `q95` 是名义 90% 预测区间，覆盖率须在独立、相关人群上测量；模型不将输出强制裁剪到 0–30。
- 合成数据验证工程流程，不能验证实际病程机制、跨人群泛化、因果解释或临床可用性。

接入自有数据时，先在本地将已获授权的数据转换为三张 CSV，明确每个字段的单位和可用时间，保留人员 ID 的前导零，并处理同日多次访视。然后运行 `prepare` 和 `check`，审查报告后重新训练。不要将真实数据、媒体、身份信息或由真实数据生成的模型提交到此仓库。适配细节见[数据格式](docs/data-format.md#真实数据适配)。

## 方法来源与引用

关系卷积依据 [Schlichtkrull 等，Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)；不规则时间处理参考 [Baytas 等，Patient Subtyping via Time-Aware LSTM Networks](https://www.kdd.org/kdd2017/papers/view/patient-subtyping-via-time-aware-lstm-networks) 及[作者仓库说明](https://github.com/illidanlab/T-LSTM)；分位数目标参考 [Koenker 与 Bassett，Regression Quantiles](https://www.jstor.org/stable/1913643)。本仓库从公开公式独立实现，作者仓库只作为方法出处，不复制其实现。具体组合与公式见[方法文档](docs/method.md)。

引用本软件请使用 [CITATION.cff](CITATION.cff)。许可全文见 [LICENSE](LICENSE)。
