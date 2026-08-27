# 审稿人 #1 第二轮回复进度

---

## 审稿意见概述

> Thanks for the detailed revision. Most of my previous concerns have been addressed: the overclaimed step-level claims are properly softened, and the added t-SNE/AUC evidence and the new Section 5.4 analysis answer my main questions. A few points remain:

第一轮三条全部 ✅ 已进 tex。第二轮新提三条,均为 minor,主要是澄清与一个小实验。

---

## 问题 1(第二轮):Table 1 数字变没解释 + C-CoT 行矛盾

- **问题**:
  1. Table 1 几乎所有数都变了(例:Standard CoT on GSM8K/LLaMA 43.80 → 71.50),但正文从没说明原版评估错在哪。
  2. response letter 称原版 C-CoT 行是 `--`(未填),但第一轮 R1 审稿时那行有数字 → 矛盾,要纠正。
- **状态**:✅ 已落地(response letter R1-Q1 段定稿;tex 不动,Table 1 终版值已在;C-CoT 矛盾按"原版有数字但口径不统一"诚实更正)
- **分析**:
  - **三处错的根因**:
    ① **GSM8K/LLaMA 旧列**(原版 line 372–374):test=42 小切分(8:1:1)+ 数据规模声称 1,000 题/数据集(原版 line 357,但实际 CSQA 数据只有 500 题,口径不实)→ 换官方 test n=200 + 真实 per-backbone 切分(Table 5 现值)。原版 Std 43.80 / SC 63.20 / C-CoT 87.00(bold) / ORM 78.20 / D-ProtoCoT 80.95 → 现版 Std 71.50 / SC 71.50 / C-CoT 78.84 / ORM 68.50 / D-ProtoCoT 80.00。**旧 Static 87 神话已破**(原版无 Static-Prototype 行;新增后 68.50 < Std 71.50)。
    ② **StrategyQA 旧列**:旧抽取器漏 `$\boxed{yes/no}$`,per-path 被误判 28.5%(真实 46.8%)。旧 Std 68.60 / SC 62.60 / C-CoT 64.40 / ORM 65.71 / D-ProtoCoT 86.20(+23.6 系坏标签假象)→ 重标 + 重跑后 Std 45.24 / SC 61.90 / C-CoT 76.81 / ORM 54.76 / D-ProtoCoT 66.67。
    ③ **CSQA/Qwen 旧列**:3-epoch + 旧切分,原版 D-ProtoCoT=87.71、SC=86.98(异常高,口径不实)→ 改 10-epoch + n=50 真切分,新值 Std 62 / SC 62 / C-CoT 78.63 / ORM 68 / D-ProtoCoT 70。CSQA/LLaMA 同步由旧 3-epoch 值(Std 68.23/SC 77.40/C-CoT 64.40/ORM 59.52/D 79.80)→ 10-epoch 新值(Std 66.67/SC 71.43/C-CoT 70.50/ORM 71.43/D 76.19)。**注**:大修中 3-epoch 重跑中间值 D=80.00(CSQA/Qwen)被弃,不进正文;最终用 10-epoch 70.00。
  - **C-CoT 行矛盾(已核实,R1 指控成立)**:`revision_notes.md` line 181/213/221 三处写"原版 C-CoT 行 `--` 未填,本轮填入真值"。但 `cas-dc-template_original.tex` line 374 实测 C-CoT 行**原版 6 格全有数字**:`64.40 & \textbf{87.00} & 64.40 & 68.20 & 75.60 & 62.70`(原版无 14B 列,所以 6 格)。R1 第一轮审的就是这版,看到的是有数字的行 → 第二轮明确点出"response letter 说原版 `--` 与我审的版本矛盾"。**第二轮 response letter 必须承认这个事实错误**,改口为"原版 C-CoT 行有数字(64.40/87.00/...),但来自非统一口径实现,本轮用统一 pipeline (`newrun/ccot_prompting.py`) 重跑,数字相应更新为新值"。
  - **C-CoT 数字变化对照(原版→现版)**:
    - L-CSQA:64.40 → 70.50
    - L-GSM8K:**87.00(bold)** → 78.84(去 bold)
    - L-SQA:64.40 → 76.81
    - Q-CSQA:68.20 → 78.63
    - Q-GSM8K:75.60 → 92.35
    - Q-SQA:62.70 → 90.22
    - 14B GSM8K:原版无此列 → 95.50(新增)
- **计划**:
  1. 正文 §4.6(Main Results 后)或 §5.6(Comparison with ORM 附近)加段"Revisions to the Originally Reported Numbers",诚实交代三处错因 + 新值 + C-CoT 重跑口径。
  2. response letter 第二轮:① **承认 C-CoT 行说法不实**(原版有数字非 `--`,改为"原版有数字但口径不统一,本轮重跑");② 逐条交代数字变更原因(test=42→200、抽取器 bug、3→10 epoch)。
- **状态**:✅ 已落地(response letter R1-Q1 段定稿 + StrategyQA 因果链按规则 4 改为只报修复前后事实不画因果 + C-CoT 矛盾按"原版有数字但口径不统一"诚实更正;**tex 不加"Revisions"段,按规则 2 发表版保持干净**)

---

### 版本 A:Response letter 草稿(英文,回信用,待写)

> 待写:对照 revision_notes.md 的格式,逐条回 R1-Q1/Q2/Q3 + R3。

---

### 版本 B:正文改动(tex 落地,待写)

> 待写:在 §4.6 或 §5.6 加段"Revisions to the originally reported numbers",诚实交代三处错因。具体措辞见分析段。

---

## 问题 2(第二轮):§4.1 vs Table 5 切分不一致

- **问题**:§4.1 说"the two backbone models share the same held-out question set",但 Table 5 显示 CSQA Qwen3-8B test=50 vs LLaMA test=42、SQA Qwen3-8B test=28 vs LLaMA test=42。两 backbone 切分大小不同 → §4.1 那句不实。还要解释为什么 train 大小也不同。
- **状态**:✅ 已落地(§4.1 line 338 改写 + Table 5 caption 补注 + response letter 定稿)
- **分析**:
  - 每个 backbone 各自生成路径(路径是 backbone-specific 的),各自 8:1:1 切分。Qwen3-8B CSQA 共 500 题(train 400/val 50/test 50),LLaMA CSQA 共 420 题(train 336/val 42/test 42)。**根本不是同一份 held-out 集**。
  - §4.1 那句"share the same held-out question set"是**事实错误**,要改。
  - Train 大小不同:各 backbone 路径池大小不同(Qwen3-8B 在 CSQA 上路径更多)。
- **计划**:
  1. §4.1 改"share the same held-out question set"那句,改为诚实表述(每个 backbone 各自 8:1:1,问题 id 分组,无跨切分泄露;但两 backbone 切分独立,大小不同)。
  2. Table 5 caption 同步补一句"each backbone has its own 8:1:1 split, sizes differ because path pools differ"。
- **状态**:✅ 已落地(§4.1 line 338 改写 + Table 5 caption 补注;response letter 含交集过滤机制诚实交代 + "within-backbone 主结论不受影响"澄清)

---

### 版本 B:正文改动(待写)

> §4.1 原文(line 338):
> > "the two backbone models share the same held-out question set so that per-model results remain directly comparable"
>
> 待改为:每个 backbone 各自 8:1:1 qid 分组切分;两 backbone 切分独立,大小不同;但均保证零跨切分泄露。

---

## 问题 3(第二轮):newline 切步未答

- **问题**:第一轮问过 newline-based step segmentation,这次说没答。换行切可能切半步或并步,影响 step-level 训练信号。要求做个"quick comparison with explicit Step k: markers"说明 robust。
- **状态**:✅ 已落地(10 epoch 跑完:newline 79.50% / step_marker 82.00% / Δ=+2.50,落 noise band;§5.3 加 Step-Delimiter Robustness 段 + response letter 定稿)
- **分析**:
  - 论文 §3.2 line 218:`segmented into M steps based on newline delimiters`。
  - 审稿人担心:换行切可能切错 → 影响 InfoNCE 对齐目标。
  - **实验设计**:
    - 基线切法:按 `\n` 切(现版做法)
    - 对比切法:按 `Step \d+:` 正则切(显式 marker)
    - 各跑一次 encoder 训练 + 评估(GSM8K/Qwen3-8B,K=10,10 epoch)
    - 比较准确率,期望 |Δ| ≤ 1~2 点 → robust
  - **第一步**:grep `newrundata/gsm8k_merged_flat.jsonl` 的 `cot` 字段,看有没有 "Step" 标记。若无 → 要么换思路(用 GPT-4 等重生成带 marker 的 CoT),要么换数据集试。
- **计划**:
  1. 写切步对比脚本(`newrun/step_seg_compare.py` 或类似)
  2. 跑两次训练(newline vs Step k:)
  3. 论文 §3.2 后或 §5.x 加一小段(可选小表)说明 robust
- **状态**:✅ 已落地(§5.3 末加 `\paragraph{Step-Delimiter Robustness.}`;实测 newline 79.50% / step_marker 82.00% / Δ=+2.50 落 §5.3 已承认的 2.50 noise band;脚本 `newrun/reviewer1_q3_segmentation.py` + JSON `newrun/reviewer1_q3_segmentation.json`)

---

### 版本 B:正文改动(待写)

> §3.2 line 218 后或 Analysis 新增一小段:
> > "We verified that the choice of step delimiter does not materially affect performance: segmenting by explicit 'Step k:' markers (when present in the generation) yields accuracy within X points of newline-based segmentation on GSM8K/Qwen3-8B (Y% vs Z%), indicating the method is robust to this choice."

---

## 总结:三条意见的解决路径

| 意见 | 能否纯文本 | 需要的实验/图 | 状态 |
|------|-----------|--------------|------|
| #1 数字变没解释 + C-CoT 矛盾 | 是(正文+letter)| 无 | ✅ 已落地(response letter 定稿;tex 不动,Table 1 终版值已在) |
| #2 §4.1 vs Table 5 切分不一致 | 是 | 无 | ✅ 已落地(§4.1 line 338 改写 + Table 5 caption 补注 + response letter 定稿,含交集过滤机制诚实交代) |
| #3 newline 切步 | 否(要跑实验) | newline vs Step k: 对比 | ✅ 已落地(10 epoch 跑完:79.50% vs 82.00%,Δ=+2.50;§5.3 加 Step-Delimiter Robustness 段 + response letter 定稿) |

---

## 附录:审稿人 #1 第二轮原话

Reviewer #1: Thanks for the detailed revision. Most of my previous concerns have been addressed: the overclaimed step-level claims are properly softened, and the added t-SNE/AUC evidence and the new Section 5.4 analysis answer my main questions. A few points remain:

1. Almost all numbers in Table 1 changed from the previous version (e.g., Standard CoT on GSM8K/LLaMA went from 43.80 to 71.50), but this is never explained. Please state what was wrong in the original evaluation. Also, the response letter says the original C-CoT row was unfilled ("--"), but it had numbers in the version I reviewed, so please correct this.

2. Section 4.1 says the two backbones share the same held-out question set, but Table 5 shows different test sizes (CSQA 50 vs 42, StrategyQA 28 vs 42). Which is correct? And why do the training set sizes differ by backbone?

3. My question about newline-based step segmentation was not answered. Newline splitting can cut a step in half or merge steps, which directly affects the step-level training signal, so a quick comparison with explicit "Step k:" markers would be enough to show the method is robust to this choice.

With these clarified I would support acceptance.
