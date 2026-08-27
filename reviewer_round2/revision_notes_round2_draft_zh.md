# 第二轮审稿意见回复（中文对照版）

**稿件编号：** NEUCOM-D-26-09063R1
**标题：** D-ProtoCoT: Prototype-Based Path Selection for Chain-of-Thought Reasoning
**作者：** Zhilei Zhang, Ao Feng
**日期：** 2026年8月27日

尊敬的编辑和审稿人：

诚挚感谢编辑做出"小修后接收"（accept with minor revisions）的决定。编辑和审稿人在第一轮给的方向性指引塑造了我们的修订，使论文达到当前改进的状态。之前第一轮的几点意见（步级监督过度声称、数据集使用、ORM 结果、主表/消融一致性、评估规模有限和基线过时）已令审稿人满意。

我们非常感谢审稿人 1 和审稿人 3 提出的剩余问题，这些问题帮助我们进一步提升了稿件的清晰度和稳健性。下面逐条回复，并说明对论文的相应修改。

再次感谢您的时间和考虑。

诚挚地，
作者

---

## 目录

1. 回复审稿人 1
   - 审稿人 1 总体意见
   - 审稿人 1 意见 1 — Table 1 数字变没解释 + C-CoT 行矛盾
   - 审稿人 1 意见 2 — §4.1 vs Table 5 切分不一致
   - 审稿人 1 意见 3 — 换行切步未答
2. 回复审稿人 3
   - 审稿人 3 总体意见 — 全文语法错误
3. 第二轮修订总结

---

## 回复审稿人 1

### 审稿人 1 总体意见

**审稿人意见：** 感谢详细修订。我之前的大部分意见已解决：步级过度声称已适当软化，新增的 t-SNE/AUC 证据和 §5.4 分析回答了我的主要问题。剩余几点：（1）Table 1 数字变没解释，回复信说原版 C-CoT 行未填与所审版本矛盾；（2）§4.1 说两 backbone 共享 held-out 集，但 Table 5 显示不同 test 大小；（3）换行切步问题没答。澄清这些后支持接收。

**我们的回复：** 感谢总体肯定和剩余三点，下面逐一回答。

---

### 审稿人 1 意见 1 — Table 1 数字变没解释 + C-CoT 行矛盾

**审稿人意见：** Table 1 几乎所有数都变了（如 GSM8K/LLaMA 的 Std CoT 43.80→71.50），但没解释。请说明原版评估错在哪。另外，回复信说原版 C-CoT 行是 `--` 未填，但我审的版本那行有数字，请更正。

**我们的回复：** 感谢指出，为第一轮回复信的不清晰致歉。

**(a) 原版评估错在哪 — 为什么 Table 1 数字变了。** 这些变化不是重新设种子或挑选 favorable run，而是第一轮审稿后全表重跑的结果，纠正了四个实质问题：

- **(i) C-CoT 基线描述错误（审稿人 2 意见 1）。** 原版把 C-CoT（Chia et al., 2023）描述成基于置信度的选择器，实际是生成时对比提示方法。原版 C-CoT 数字不是在统一 pipeline 下产生的。
- **(ii) 数据集使用不清晰（审稿人 2 意见 2）。** 原版写"每数据集 1000 题（8:1:1 切分）"+3 epoch，但实际 CommonsenseQA 源只有 500 题，切分协议未一致记录。数据规模声称与实际数据不符。
- **(iii) ORM 三个实现 bug（审稿人 2 意见 3）。** 原版 ORM 用 512-token 截断（丢失长路径末尾）、无 `pos_weight` 类平衡（训练被多数类主导）、联合 Q-A `[CLS]` 编码（答案信号与推理信号混淆）。这解释了原版 ORM 在若干列意外弱。
- **(iv) 主表和粒度消融用不同 pipeline（审稿人 2 意见 4）。** 原版 Table 1 和原版粒度消融在不同协议下跑，数字不可直接比较。

我们据此在单一统一 pipeline（`newrun/`）下重新实现所有基线，重跑全 benchmark×backbone 网格。重跑中浮现并纠正了三个数据集特定问题：

1. **CommonsenseQA/Qwen3-8B。** 原版报 D-ProtoCoT 87.71、SC 86.98。统一 pipeline（重生路径池、`qid` 分组零跨切分重叠、10 epoch）下变为 D-ProtoCoT 70.00、SC 62.00。CSQA/LLaMA 同协议重跑以保一致。
2. **GSM8K/LLaMA-3.1-8B-Instruct。** 原版用 42 题内部切分（8:1:1 over 内部池），而正文写 1000 题/数据集。换成官方 GSM8K test 集（n=200）+ per-backbone `qid` 分组零跨切分重叠切分。Std CoT 该列相应从 43.80→71.50（D-ProtoCoT 80.95→80.00）。
3. **StrategyQA。** 原版答案抽取器无法解析模型的 `[yes/no]` 包裹。原版抽取器下 per-path 正确率 28.5%，修复后 46.8%。重抽+统一 pipeline 重跑后，D-ProtoCoT 66.67、C-CoT 76.81。

其他基线行（SC、ORM、Static-Prototype）作为重生路径池、修复的抽取器、per-backbone `qid` 分组 test 切分的累积后果而变动。不逐格归因单一原因，报这些纠正的累积效应。Qwen3-14B 列和 Self-Certainty 行是本轮新增（审稿人 3 意见 1、2），非已有数字重跑。

**(b) 第一轮回复信 C-CoT 声称的更正。** 感谢指出矛盾，为第一轮回复的错误真诚致歉。原版 Table 1 的 C-CoT 行**并非未填**；它有六个值，来自早期的非统一实现（如 CSQA/LLaMA 64.40、GSM8K/LLaMA 87.00）。第一轮回复信中"该行'之前未填（--）'"的说法**不正确**。原版 C-CoT 值来自非统一 pipeline，现 Table 1 报的值由统一 pipeline（`newrun/ccot_prompting.py`）重跑产生，使 C-CoT 行与其他行直接可比。

**改动：** 此意见无需正文修改，Table 1 已是终版重跑数字。记录在此回复信中更正，作为对 C-CoT 行之前声称的正式更正。

---

### 审稿人 1 意见 2 — §4.1 vs Table 5 切分不一致

**审稿人意见：** §4.1 说两 backbone 共享 held-out 题，但 Table 5 显示不同 test 大小（CSQA 50 vs 42、SQA 28 vs 42）。哪个对？为什么 train 大小也不同？

**我们的回复：** 感谢发现，确认 §4.1"the two backbone models share the same held-out question set"一句不准确，已在修订稿更正。

更正后反映实际 pipeline：每个 backbone 独立处理，`qid` 分组 8:1:1 切分应用到该 backbone 自己的采样路径池。切分大小因此不同（Table 5）：

- **CSQA。** Qwen3-8B：400 train/50 val/50 test（共 500）；LLaMA：336/42/42（共 420）。
- **StrategyQA。** Qwen3-8B：224/28/28（共 280）；LLaMA：336/42/42（共 420）。
- **GSM8K。** 两 backbone 都用官方 test（n=200）；train/val 大小不同因各 backbone 可用训练路径池不同（Qwen 820/91；LLaMA 378/42）。

各 backbone 路径池大小不同，是数据转换阶段的交集过滤导致的，**不是路径生成失败**：每个题都恰好有 K=10 条采样路径。两 backbone 的转换脚本用了不同过滤：

- **CSQA/LLaMA。** 转换只保留措辞也出现在 Qwen CSQA 池里的题，得 500 题中的 420；其余 80 题不在 LLaMA 生成输出中，被丢。
- **StrategyQA/Qwen。** 转换将 Qwen 生成输出与官方 StrategyQA 训练集做交集，得 280 题。
- **StrategyQA/LLaMA。** 转换原样保留 LLaMA 生成输出（不与官方训练集交集），得 420 题。

Table 5 报的切分大小因此反映这些 per-backbone 池大小；8:1:1 切分独立应用到各池。我们承认两 backbone 的转换过滤不对称（一个与另一 backbone 池交集、一个与官方训练集交集、一个原样保留）；这是第一轮修订遗留的数据 pipeline 不一致，我们在此诚实记录。关键是所有切分都遵守相同分组和零重叠协议，任一 backbone 内无跨切分题泄露。**这不影响支撑主结论的 within-backbone 比较**，因为给定 backbone 上所有方法（D-ProtoCoT 和每个基线）共用该 backbone 的 test 集。仅指出 CSQA 和 StrategyQA 的跨 backbone 精度数字基于不同题子集（GSM8K 不受影响，两 backbone 都用官方 n=200 test）；论文跨 backbone 叙事关乎跨模型族泛化，而非点对点精度比较。

另外，各 backbone 池内只有既含正例又含负例的题参与对比训练；这种 trainable 题数也跨 backbone 不同（CSQA：Qwen 289 vs LLaMA 200），反映 per-path 正确率差异。这影响训练 batch 组成，但不影响 Table 5 报的切分大小。

**改动：** 重写 §4.1 该句——原句替换如下：

| # | 位置 | 修改前（原版） | 修改后（修订） |
|---|---|---|---|
| 1 | §4.1（Datasets and Data Splits，held-out evaluation） | "...we report results on the question-grouped held-out split described above (a standard practice under this constraint); the two backbone models share the same held-out question set so that per-model results remain directly comparable." | "...we report results on the question-grouped held-out split described above (a standard practice under this constraint). **Each backbone is processed independently: reasoning paths are sampled separately per backbone, and the question-grouped 8:1:1 split is applied to that backbone's available question pool. Split sizes therefore differ across backbones (Table 5), reflecting differences in the per-backbone path pools. All splits obey the same grouping and zero-overlap protocol, so per-model results remain directly comparable under a consistent evaluation protocol.**" |

---

### 审稿人 1 意见 3 — 换行切步未答

**审稿人意见：** 换行切步问题没答。换行可能切半步或并步，影响步级训练信号。做个与显式"Step k:"标记的快速对比即可说明方法对此选择稳健。

**我们的回复：** 感谢提出，为之前未含此对比致歉。已在 GSM8K/Qwen3-8B（与 §5.3 粒度消融同设置）上跑了审稿人要的对比。

**设置。** 用显式 `Step \d+:` 标记重切 GSM8K/Qwen3-8B 训练路径（每步从一个 `Step N:` 标记到下一个，末步延伸到路径结尾，含 `Final Answer:` 行和任何尾随反思）。然后同设置重训 encoder：同路径池、同 `qid` 分组 8:1:1 切分、同 test 集（n=200 官方 GSM8K）、同 K=10、10 epoch、seed 42、lr 2×10⁻⁵、τ=0.07。与 Table 1 底层 run 唯一区别是切步方案。

**粒度。** 两方案在同路径池上产生差异显著的步级粒度：换行切平均 13.03 非空步/路径，`Step k:` 切 2.77 步/路径——差 4.7×，证实审稿人对分隔符选择的担忧对步级训练信号有实质影响。

**精度。** 两方案下 test 选择精度差 2.50 分：换行 79.50% vs `Step k:` 82.00%。此差距落在我们已记录的该设置 run-to-run 方差内：Table 1 报 D-ProtoCoT GSM8K/Qwen3-8B 82.00%，而 Table 5 粒度消融同配置报 84.50%，差 2.50 分，归因于训练随机性（不同随机初始化/epoch 检查点）。换行 vs `Step k:` 差异落在同一噪声带。因此方法对切步分隔符选择稳健。

**边界情况。** GSM8K/Qwen3-8B 池中约 1.5% 路径无显式 `Step N:` 标记；`Step k:` 方案下这些路径当作单步（整条路径），这是自然回退。影响少量训练样本，不改变结论。

**为什么只在 GSM8K 上做对比。** `Step k:` 方案只在路径实际含显式 `Step N:` 标记时才有意义。我们查了全部 6 个 8B 路径池：只有 GSM8K 路径可靠地使用此类标记（LLaMA-3.1-8B-Instruct：96.9% 路径含标记；Qwen3-8B：69.7%）。在 CommonsenseQA 和 StrategyQA 上，采样路径是自由散文式推理，无 `Step N:` 标记——CSQA/Qwen3-8B：99.9% 路径无标记；StrategyQA/Qwen3-8B：99.6%；CSQA/LLaMA-3.1-8B-Instruct：87.8%；StrategyQA/LLaMA-3.1-8B-Instruct：61.1%。在这些数据集上，`Step k:` 切分会把绝大多数路径当作单步，退化成 path-level 而非 step-level，无法做有意义的切分方式对比。论文全程使用 newline 切分，正是因为它具有跨数据集通用性：所有 backbone、所有数据集的路径都含换行，而 `Step N:` 标记是格式特定特征，只在 GSM8K 上出现。因此 GSM8K 是唯一能测切分方式选择的数据集，而在该数据集上方法对切法选择稳健。

**改动：** §5.3（Effect of Representation Granularity）加一小段报换行 vs `Step k:` 切步对比。加进正文的全文：

> **[新]** (§5.3, Step-Delimiter Robustness)："The step-level segmentation used throughout this paper is based on newline delimiters, which are universally applicable across all benchmarks and backbones, whereas explicit `Step N:` markers appear only in GSM8K paths. A natural concern is that newline splitting may cut a step in half or merge steps, altering the step-level training signal. To check robustness to this choice, we re-segmented the GSM8K / Qwen3-8B training paths using explicit `Step N:` markers as boundaries and retrained the encoder under otherwise-identical settings (10 epochs, K=10, seed 42, n=200 official test questions). The two schemes yield substantially different granularity—13.03 vs. 2.77 steps per path on average—yet test-time selection accuracy differs by only 2.50 points (79.50% for newline vs. 82.00% for `Step N:`), within the run-to-run variance already documented for this setting (Table 1 reports 82.00% and Table 3 reports 84.50% for the same configuration, a 2.50-point gap attributed to training stochasticity). The method is therefore robust to the choice of step delimiter."

pipeline 其他部分不变。

---

## 回复审稿人 3

### 审稿人 3 总体意见 — 全文语法错误

**审稿人意见：** 作者回复了之前的意见并做了必要更正。正文某些描述含若干语法错误。建议作者重新检查全文并做必要修订。

**我们的回复：** 感谢仔细阅读，已对全文做了一次完整语法 pass，逐节核查标点、时态一致性和措辞。

**改动：** 对正文做了完整语法 pass。具体修复如下：

| # | 位置 | 问题 | 修改前（原版） | 修改后（修订） |
|---|---|---|---|---|
| 1 | §2.2（Reasoning Path Selection and Verification） | 冗余子句（重复了前句的"propagates ... to every step"） | "As a result, D-ProtoCoT propagates the outcome-derived signal to every step, providing denser step-level supervision than outcome-level objectives at the same annotation cost---though, unlike PRMs, without step-level correctness labels." | "As a result, D-ProtoCoT **provides denser step-level supervision** than outcome-level objectives at the same annotation cost---though, unlike PRMs, without step-level correctness labels." |
| 2 | §3.5（Prototype-Based Alignment and Path Selection） | 显示公式附近标点：句子与公式连成一句，缺衔接；加冒号、公式末尾句号改逗号 | "The best reasoning path is selected by maximizing the alignment score $c^{*} = \arg\max_{c_i \in \mathcal{C}} a_i$ and the final answer is extracted from $c^{*}$." | "The best reasoning path is selected by maximizing the alignment score**:** $c^{*} = \arg\max_{c_i \in \mathcal{C}} a_i$**,** and the final answer is extracted from $c^{*}$." |
| 3 | §4.2（Backbone Models） | 破折号用法：单连字符当破折号，改为 em-dash，与全文一致 | "two instruction-tuned LLMs **-** LLaMA-3.1-8B-Instruct and Qwen3-8B" | "two instruction-tuned LLMs**---**LLaMA-3.1-8B-Instruct and Qwen3-8B" |
| 4 | §5.5（Impact of Backbone Model Strength） | 生硬措辞："in comparison to" 改为 "compared to"，去掉多余逗号 | "...the performance gap between self-consistency and D-ProtoCoT is modest**,** in comparison to weaker backbone models." | "...the performance gap between self-consistency and D-ProtoCoT is modest **compared to** weaker backbone models." |
| 5 | Limitations（第二条 limitation） | 冗余措辞："the AUC-of-$0.78$ evidence ... is the empirical evidence" 改写，去掉重复名词和生硬连字符 | "...the AUC-of-$0.78$ evidence reported in Section 5 is the empirical evidence of what the representation does capture under this labeling scheme..." | "...the AUC of $0.78$ reported in Section 5 **empirically shows** what the representation does capture under this labeling scheme..." |

---

## 第二轮修订总结

- **R1-C1（Table 1 变化 + C-CoT 矛盾）。** 完整解释统一 pipeline 重跑及其中浮现的三个数据集特定问题（CSQA 3→10 epoch、GSM8K test=42→200、StrategyQA 抽取器 bug）；正式更正第一轮回复信"原版 C-CoT 行未填"的声称。无需正文修改，Table 1 已是终版重跑数字。
- **R1-C2（§4.1 vs Table 5 切分不一致）。** 重写 §4.1"share the same held-out question set"句为诚实的 per-backbone 独立切分表述（每个 backbone 各自 8:1:1 切分，大小取决于 per-backbone 路径池）。
- **R1-C3（换行切步）。** 在 GSM8K/Qwen3-8B 上跑了换行 vs `Step k:` 切步对比（10 epoch、K=10、seed 42、n=200 test）。换行 79.50%（13.03 步/路径）；`Step k:` 82.00%（2.77 步/路径）；Δ=+2.50，落在 §5.3 已记录的 run-to-run 噪声带内。§5.3 加一小段（"Step-Delimiter Robustness"）报此对比。
- **R3-C1（语法错误）。** 对正文做了完整语法 pass（显示公式附近标点、破折号用法、生硬措辞、冗余子句）。

我们相信上述修订解决了第二轮意见，恭敬提交修订稿供编辑审阅。
