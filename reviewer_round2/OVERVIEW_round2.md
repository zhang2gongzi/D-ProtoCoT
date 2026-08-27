# 第二轮审稿回复进度(OVERVIEW)

> 跨审稿人全局进度。详情见各 `round2_reviewerN_progress.md`。最后更新:2026-08-27(R1 三条 + R3 全部落地)。
> Manuscript ID:NEUCOM-D-26-09063R1
> 状态:**Minor revision(条件接收)**　Deadline:**2026-09-08 23:59**
> 图例:✅ 已进 tex / 完成　🟡 数据部分就绪,待收尾　🔴 硬卡点　⚪ 暂缓

---

## 总览

| 审稿人 | 条数 | 状态 |
|---|---|---|
| Reviewer #1 | 3 | ✅ 全部落地(Q1 response letter;Q2 tex §4.1;Q3 tex §5.3+实验) |
| Reviewer #2 | 0(缺席) | ⚪ 第二轮无意见 |
| Reviewer #3 | 1(全文语法) | ✅ 落地(语法 pass + 删中文注释) |

---

## Reviewer #1(3 条,第二轮新意见)

| 条 | 问题 | 状态 |
|---|---|---|
| 1 | Table 1 数字变没解释 + response letter 称原版 C-CoT 行 `--` 矛盾 | ✅ response letter 定稿(四条 R2 触发 + 三数据集原因 + C-CoT "incorrect" 更正);tex 不动 |
| 2 | §4.1 说两 backbone 共用 held-out 集,但 Table 5 显示 CSQA 50 vs 42、SQA 28 vs 42 | ✅ §4.1 line 338 改写 + response letter(含交集过滤机制 + within-backbone 不影响主结论) |
| 3 | newline 切步未答(第一轮问过),要和 "Step k:" 对比说明 robust | ✅ 10 epoch 跑完(newline 79.50% / step_marker 82.00% / Δ=+2.50);§5.3 加 Step-Delimiter Robustness 段 |

---

## Reviewer #3(1 条,语法)

| 条 | 问题 | 状态 |
|---|---|---|
| 1 | 全文语法错误,通读改 | ✅ 删 3 块中文编辑注释 + 修 5 处语法(§3.5 公式标点 / §4.2 破折号 / §5.4 措辞 / §2.2 冗余 / §Limitations 冗余) |

---

## 提交清单(Deadline 2026-09-08)

| 项 | 状态 |
|---|---|
| Response letter(`response_letter_round2_draft.md`,按第一轮 `revision_notes.md` 格式) | ✅ 定稿(R1 三条 + R3 一条;待提交时改日期) |
| 修订 tex(`cas-dc-template_round2_revision.tex`,本轮终版) | ✅ R1-Q2 §4.1 + R1-Q3 §5.3 段 + R3 删注释/修语法 |
| 红字 diff tex(`cas-dc-template_round2_red.tex`) | ✅ 8 处红标覆盖 response letter 全部 tex 改动 |
| 修订 PDF | ⚪ 待编译(用户本地捯饬) |
| LaTeX 源 + 图文件打包 | ⚪ 待打包 |
| 作者简介 + 照片(tex line 679–685 已有占位) | 🟡 需核对照片文件是否齐全 |
| **⚠️ 作者名单确认**(一旦接收不能再改) | 🟡 需核对 |

---

## 工作优先级与顺序(全部完成 ✅)

1. ✅ **R1-Q2**:§4.1 line 338 改写(response letter 含交集过滤机制)
2. ✅ **R1-Q1**:response letter 纠错(四条 R2 触发 + 三数据集原因 + C-CoT "incorrect")
3. ✅ **R1-Q3**:切步对比实验跑完(newline 79.50% vs step_marker 82.00%)+ §5.3 加段
4. ✅ **R3**:全文语法 pass + 删 3 块中文注释
5. ⚪ **提交材料**:编 PDF + 打包 + 核对作者/照片 + 提交(用户侧)

---

## 关键文件位置

- **本轮终版 tex:`reviewer_round2/cas-dc-template_round2_revision.tex`**(683 行,R1-Q2 §4.1 + R1-Q3 §5.3 段 + R3 删注释/修语法;本轮提交用此文件)
- **本轮红字 diff:`reviewer_round2/cas-dc-template_round2_red.tex`**(8 处红标覆盖 response letter 全部 tex 改动声明)
- **本轮 response letter:`reviewer_round2/response_letter_round2_draft.md`**(按第一轮 `revision_notes.md` 格式,R1 三条 + R3 一条全定稿)
- **R1-Q3 实验脚本/数据:`newrun/reviewer1_q3_segmentation.py` + `newrun/reviewer1_q3_segmentation.json` + `newrun/logs/r1q3_segmentation_*.log`**
- **根目录 `cas-dc-template.tex`**:8/13 投稿原版(已回退干净,不动,作 R1-final 参照)
- 红字 diff 脚本:`reviewer_round2/make_round2_red_diff.py`(OLD=R1-final vs NEW=round2_revision)
- 原始版(投稿被审):`../reviewer/cas-dc-template_original.tex`(590 行,未修订)
- 第一轮 response letter:`../response_letter_v2.pdf` + `reviewer/revision_notes.md`(line 130 是 3.6M blob,跳过)
- 第一轮进度记录:`reviewer/OVERVIEW.md` + `reviewer/reviewer{1,2,3}_progress.md`
- 第二轮审稿意见原文:`reviewer_comments_round2.md`(本文件夹)

---

## 口径红线(照 cas-dc-template.tex 实文核对,2026-08-26)

### 主表(Table 1,line 397–403)真值
| 行 | LLaMA-CSQA | LLaMA-GSM8K | LLaMA-SQA | Qwen-CSQA | Qwen-GSM8K | Qwen-SQA | 14B-GSM8K |
|---|---|---|---|---|---|---|---|
| Std CoT | 66.67 | 71.50 | 45.24 | 62.00 | 75.00 | 60.71 | 93.50 |
| SC | 71.43 | 71.50 | 61.90 | 62.00 | 77.50 | 67.86 | 97.00 |
| C-CoT | 70.50 | 78.84 | 76.81 | 78.63 | 92.35 | 90.22 | **95.50** |
| Static-Proto | 71.43 | 68.50 | 52.38 | 62.00 | 81.00 | 64.29 | 97.00 |
| ORM | 71.43 | 68.50 | 54.76 | 68.00 | **92.00** | 60.71 | 96.50 |
| **D-ProtoCoT** | **76.19** | **80.00** | **66.67** | **70.00** | 82.00 | **71.43** | **97.50** |

**关键事实**(勿犯):
- **D-ProtoCoT ≥ SC 在全部 7 列**(LLaMA +4.76/+8.50/+4.77、Qwen +8.00/+4.50/+3.57、14B +0.5);但 Q-SQA +3.57≈1 题(n=28)、14B +0.5 饱和 → tex 仍用"in most settings"保守限定(line 67/104/408 三处一致)
- **GSM8K/Qwen3-8B:ORM 92 > D-ProtoCoT 82**(line 402,ORM 数字加粗,非 D-ProtoCoT)。GSM8K/Qwen3-14B:D-ProtoCoT 97.50 > ORM 96.50(14B 反超 ORM)
- **Static-Prototype 从不超过 D-ProtoCoT 任何数据集**(line 417);LLaMA-GSM8K 上 Static 68.50 < Std 71.50(line 417,跌破基线)
- **C-CoT 行(line 399)7 格全有数字**,14B GSM8K = 95.50(已填,非 `--`);response letter 第二轮**严禁再写**"原版 C-CoT 行 `--`"(R1 已点矛盾)
- **ORM 在 LLaMA-CSQA 过拟合**:val_loss 0.63→2.0、AUROC 0.559(近随机),D-ProtoCoT 76.19 > ORM 71.43(line 548)— "两区间叙事"(饱和 ORM 强 / 难任务 D-ProtoCoT 稳)

### Q2 小表(Table 2,line 425–439,GSM8K/Qwen3-8B)
- SC 77.50 / Self-Certainty(Kang logprob)77.00 / Self-Certainty-BERT 81.00 / **D-ProtoCoT 82.00**
- 正文(line 422–423)**只对比 Kang 版**(D-ProtoCoT +5.0 over Kang);Self-Certainty-BERT 进表但**正文不写超过**(主表那次 run 的 Self-Certainty-BERT log 未单独存)
- **两个 Self-Certainty 别混**:Kang(logprob)=77.00(可写超过 +5);BERT(embedding)=81.00(进表但不写超过对比)

### headline 锁定
- 14B GSM8K D-ProtoCoT = **97.50**(run.py main;诊断跑 97.00 不进正文)
- 8B GSM8K/Qwen3-8B 主表 = **82.00**(主表 run.py main;诊断跑 mixed_question_analysis FULL 81.00 视作 ~1 点方差)

### 三处原版错因(R1-Q1 要写进正文,已核实 original.tex line 372–374)
1. **GSM8K/LLaMA 旧列**:test=42 小切分(8:1:1)+ 原版声称"1,000 题/数据集"(line 357,但实际 CSQA 数据只 500 题,口径不实)→ 换官方 test n=200 + 真实 per-backbone 切分。原版 Std 43.80/SC 63.20/C-CoT 87.00(bold)/ORM 78.20/D 80.95 → 现 Std 71.50/SC 71.50/C-CoT 78.84/ORM 68.50/D 80.00
2. **StrategyQA 旧列**:旧抽取器漏 `$\boxed{yes/no}$`,per-path 误判 28.5%(真实 46.8%)→ 重标 + 重跑。原版 Std 68.60/SC 62.60/C-CoT 64.40/ORM 65.71/D 86.20(+23.6 系坏标签假象)→ 现 Std 45.24/SC 61.90/C-CoT 76.81/ORM 54.76/D 66.67
3. **CSQA/Qwen 旧列**:3-epoch + 旧切分,原版 D=87.71/SC=86.98(异常高)→ 10-epoch + n=50 真切分,新 Std 62/SC 62/C-CoT 78.63/ORM 68/D 70。CSQA/LLaMA 同步:旧 3-epoch 值(Std 68.23/SC 77.40/C-CoT 64.40/ORM 59.52/D 79.80)→ 10-epoch 新值(Std 66.67/SC 71.43/C-CoT 70.50/ORM 71.43/D 76.19)。**注**:大修中 3-epoch 重跑中间值 D=80.00(CSQA/Qwen)被弃,不进正文;最终用 10-epoch 70.00

### 其他关键证据(若被追问要拿得出)
- **AUC 0.78**:alignment 预测 path correctness(GSM8K 200 题 K=10);出现在 Figure 2 caption(line 454)、§5.1(line 465)、§5.4(line 509)、§5.6(line 556)四处
- **granularity 消融**(Table,line 492–505,GSM8K/Qwen3-8B):Path/Path 78.50 / Step/Step 84.00 / **Step/Path 84.50**(proposed);step 训练 +5.5、path 选择 +0.5
- **leakage 三模式**(§5.7,line 561–566):full ≈ mask ≫ qa_only(Static 81.0/81.5/74.5)
- **14B 饱和分析**(§4.6,line 419–420):mixed 比例 8B 63.0% → 14B 3.0%,per-path 74.9%→96.6%;8B MIXED 子集(n=126)D-ProtoCoT 75.40 > SC 69.84(+5.56)

### 禁止
- **禁止编造数字**(曾拒把 ORM 92 改 85/86;无 log 支撑 = 学术不端)
- **不再强凑统一 X**:两区间叙事已落地,别再回去种子重跑
- **禁止挑"最适合填表的数"**:跑出多少填多少

