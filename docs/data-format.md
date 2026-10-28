# 数据格式（schema v1）

本文记录 v0.1.0 的实现接口与约束。所有 CSV 使用表头；日期为 ISO `YYYY-MM-DD`，数值缺失用空单元格。`person_id`、`visit_id`、`session_id` 按字符串读取，`0001` 不应转换为 `1`。不要以 `0`、`-999` 或字符串 `NA` 代替缺失数值。

## visits.csv

每行是一次访视。必需列顺序如下（读取时以列名识别）：

```csv
person_id,visit_id,visit_date,session_id,score_available_date,action_1,action_2,action_3,action_4,action_5,action_6,score,assessment_1,assessment_2,background_1,background_2,background_3,condition_1,condition_2
0001,v001,2025-01-01,am,2025-01-03,0,1,2,3,4,5,24,1,2,0,1,2,1,0
0001,v002,2025-04-01,am,2025-04-01,1,,2,3,4,5,23,1,2,0,1,2,1,0
```

| 字段 | 约束或含义 |
| --- | --- |
| person_id | 人员标识；人员之间不共享训练/验证/测试身份 |
| visit_id | 与人员标识共同确定一次访视，供其他表关联 |
| visit_date | 访视日期，决定时间排序与历史边界 |
| session_id | 同日多次访视的区分标识；不能靠猜测消除关联歧义 |
| score_available_date | 分数实际可用日期；有分数时不得早于访视日期，无分数时留空 |
| action_1…action_6 | 六个数值特征 |
| score, assessment_1, assessment_2 | 三个评估特征；score 有值时须有限且在 0–30 内 |
| background_1…background_3 | 三个背景特征 |
| condition_1, condition_2 | 两个状态特征 |

其他特征假定在 `visit_date` 已可用。若真实系统存在其他字段延迟，适配器必须先按预测起点屏蔽，或扩展可用时间契约后再使用。

模型的 14 维特征严格按上述四组拼接：6 action、3 assessment、3 background、2 condition。分数的 0–30 仅为虚构示例尺度，不对应 MMSE/MoCA。其余字段没有内置单位或临床定义，适配时须自行记录映射。

## assets.csv

```csv
person_id,visit_id,modality,path
0001,v001,audio,media/0001/v001.wav
0001,v002,video,media/0001/v002.mp4
```

标准列为 `person_id,visit_id,modality,path`。当没有 `visit_id` 时，可提供 `visit_date` 与 `session_id`，让 `prepare` 按人员、日期和必要的会话标识关联；若候选不唯一则报错，不猜测。路径必须是安全的相对路径，不能是绝对路径或包含 `..` 的目录穿越路径。

媒体仅作为索引，不参与此版本的模型输入。检查分别报告索引有效性和文件是否存在；合成文件路径可以不存在，并在来源说明中明确标记为虚构。请勿为了通过检查生成假音视频。

## targets.csv

```csv
person_id,origin_visit_id,target_date,target_score
0001,v002,2025-06-30,22
0001,v002,2025-09-28,21
0001,v002,2026-04-01,20
```

每行是一个目标日期。`origin_visit_id` 必须关联本人的访视；`target_date` 必须晚于该起点。`target_score` 用于监督训练与评估，预测时可以留空或省略整列，且预测不读取其数值。标准合成数据使用起点之后 90/180/365 天；实际 horizon 由目标日期减起点日期计算。

## 时间、缺失与泄漏边界

对每个目标，历史包含本人 `visit_date < origin_date` 的访视，以及 `origin_visit_id` 指定的起点访视，按规范时间顺序保留。同一天的其他会话即使有不同 `session_id` 也排除，因为仅凭日期不能确定其发生在预测起点之前。`session_id` 用于消除记录关联歧义，不表示时刻或先后。

若某次历史 `score_available_date > origin_date`，该分数视为缺失；基线为历史中最新的已可用原始分数。至少须有一个已可用分数，否则无法定义基线并明确报错。未来访视、未来特征及目标标签都不能成为历史输入。

准备后的一个样本包含：

| 值 | 形状 / 含义 |
| --- | --- |
| x | `float32[T,14]`，训练历史统计量标准化后的值 |
| missing | `bool[T,14]`，`True` 表示缺失；有效零值对应 `False` |
| dt_days | `float32[T]`，首项为 0，其后为相邻保留访视之间的天数 |
| baseline | 最新已可用原始分数，不是标准化分数 |
| horizon_days | 目标日期距预测起点的天数，区别于历史间隔 |
| person_id / origin_visit_id / origin_date / target_date | 关联与审计标识 |
| target_score | 仅在需要标签时读取和提供 |

预处理统计量只从训练人员在各自起点已知的历史特征拟合，并保存供验证、测试和后续预测复用。验证/测试特征、未来访视和目标标签不参与拟合。批处理 padding 由序列长度单独控制，不属于真实缺失访视。

## 真实数据适配

1. 在本地建立字段映射，确认单位、分数尺度、访视标识与实际可用日期；禁止直接把不同量表分数混作本例 `score`。
2. 分离历史访视、媒体索引和目标标签，确保无身份信息或未来信息混入数值特征。保留同日会话区分依据。
3. 输出三张 CSV，先运行 `ctp prepare --input <raw> --output <new-directory>`，再运行 `ctp check --data <prepared>`。检查错误、歧义、缺失比例与文件存在性报告。
4. 使用新的模型输出目录训练；后续预测复用该目录保存的预处理器与划分。无标签预测可省略 `target_score`。
5. 数据适配不等于效果验证。真实人群需要独立评估、量表适配和校准检验；本仓库不提供相关实证结论。

仓库只接收源代码、合成样例与文档。真实表、媒体、个人标识和真实数据衍生模型应保留在受控环境中。
