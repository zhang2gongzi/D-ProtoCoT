# 第二轮审稿意见(NEUCOM-D-26-09063R1)

> 收到日期:2026-08-26
> Deadline:2026-09-08 23:59
> 状态:**Minor revision(条件接收)** — 改完回投即正式接收
> 第一轮 ID:NEUCOM-D-26-09063 → 第二轮 ID:NEUCOM-D-26-09063R1

---

## 编辑来信(节选)

Dear Dr. Feng,

Please find below the referee reports. Based on these and the corresponding recommendation of the associate editor, I am pleased to inform you that your paper

**D-ProtoCoT: Prototype-Based Path Selection for Chain-of-Thought Reasoning**
Manuscript number: **NEUCOM-D-26-09063R1**

can be accepted for publication in Neurocomputing.

Therefore, I would very much like to invite you to revise your paper, seriously taking into account the comments of the reviewers, and to resubmit your revised version by **Sep 08 2026 11:59PM (mm/dd/yy)**. Any revision received after that may be treated as a new submission.

To submit your revision, go to http://ees.elsevier.com/neucom/ and login as an Author. You will see a menu item call Submission Needing Revision.

### Revised material should consist of
- your response to the reviewers' comments (to be uploaded as "Revision notes")
- the revised PDF of the manuscript
- the source files that have been used to prepare it (source files in LaTeX or Word, as well separate figure files; these will be used for the eventual typesetting of the paper)
- and finally, biographies and pictures of all authors

### ⚠️ 作者名单
**Please double check the author names provided in the submission and make sure to indicate any authorship related changes in the revision. Once a paper is accepted, we do not accept any changes to the author list unless explicit approval is given from co-authors and respective editor handling the submission; this may cause a significant delay in publishing your manuscript. Therefore, please make sure that you include the correct author list in the revised text of your manuscript.**

---

## Reviewer #1

> Thanks for the detailed revision. Most of my previous concerns have been addressed: the overclaimed step-level claims are properly softened, and the added t-SNE/AUC evidence and the new Section 5.4 analysis answer my main questions. A few points remain:

### R1-Q1. Table 1 数字变没解释 + C-CoT 行矛盾

Almost all numbers in Table 1 changed from the previous version (e.g., Standard CoT on GSM8K/LLaMA went from 43.80 to 71.50), but this is never explained. Please state what was wrong in the original evaluation. Also, the response letter says the original C-CoT row was unfilled ("--"), but it had numbers in the version I reviewed, so please correct this.

### R1-Q2. §4.1 vs Table 5 切分不一致

Section 4.1 says the two backbones share the same held-out question set, but Table 5 shows different test sizes (CSQA 50 vs 42, StrategyQA 28 vs 42). Which is correct? And why do the training set sizes differ by backbone?

### R1-Q3. newline 切步未答

My question about newline-based step segmentation was not answered. Newline splitting can cut a step in half or merge steps, which directly affects the step-level training signal, so a quick comparison with explicit "Step k:" markers would be enough to show the method is robust to this choice.

### R1 结论

With these clarified I would support acceptance.

---

## Reviewer #2

第二轮**缺席**(无意见)— 对第一轮回复满意。

---

## Reviewer #3

> The authors responded to the previous comments and made the necessary corrections to the manuscript. Some of the descriptions in the manuscript contain certain grammatical errors. It is suggested that the authors recheck the entire manuscript and make the necessary revisions.

### R3-Q1. 全文语法

Recheck the entire manuscript for grammatical errors; make necessary revisions.

---

## 待办优先级(作者自行排)

1. **R1-Q1**:正文加段说明 Table 1 数字为何变(三处错:test=42 小切分 → 官方 test;StrategyQA 旧抽取器漏 `$\boxed{yes/no}$`;CSQA 3-epoch 欠训 → 10-epoch);response letter 纠正 C-CoT 行 `--` 矛盾说法
2. **R1-Q2**:§4.1 改掉"share the same held-out question set"那句(事实错误,两 backbone 各自生成路径、各自 8:1:1 切分,大小不同);Table 5 caption 同步说明
3. **R1-Q3**:写个对比实验脚本,用显式 "Step k:" marker 重切 GSM8K,跑一次 encoder 训练 + 评估,说明 robust
4. **R3-Q1**:全文语法扫一遍(可借助工具;同时清 tex 里残留的中文注释)
5. **提交材料**:response letter(revision notes 上传)+ 修订 PDF + LaTeX 源 + 图 + 作者简介+照片(已有,需核对)+ **作者名单确认**

---

## 关键文件位置

- 原版 tex(投稿被审):`reviewer/cas-dc-template_original.tex`
- 工作版 tex(本轮要改的):`cas-dc-template.tex`
- 终版红字 diff(上一轮):`cas-dc-template_red.tex`
- 第一轮 response letter:`response_letter_v2.pdf` + `reviewer/revision_notes.md`
- 第一轮进度记录:`reviewer/OVERVIEW.md` + `reviewer/reviewer1_progress.md` / `reviewer2_progress.md` / `reviewer3_progress.md`
