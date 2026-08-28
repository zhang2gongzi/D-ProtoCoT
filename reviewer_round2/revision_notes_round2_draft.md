# Response to Reviewers' Comments

**Manuscript ID:** NEUCOM-D-26-09063R1  
**Title:** D-ProtoCoT: Prototype-Based Path Selection for Chain-of-Thought Reasoning  
**Authors:** Zhilei Zhang, Ao Feng

Dear editors and reviewers,

We sincerely thank the editors for the decision to accept the paper with minor revisions. The direction provided by the editor and the reviewers in the first round shaped our revision and brought the paper to its current, improved state. We are glad that those Round-1 concerns (overclaimed step-level supervision, dataset usage, ORM results, main/ablation consistency, and the limited evaluation scale and outdated baselines) have been satisfactorily addressed.

We greatly appreciate the remaining points raised by Reviewer 1 and Reviewer 3, which have helped us further improve the clarity and robustness of the manuscript. Below, we provide a point-by-point response to these comments and detail the corresponding revisions made to the paper.

Thank you again for your time and consideration.

Sincerely,  
The Authors

---

## Contents

1. Response to reviewer 1
   - reviewer 1's overall comment
   - reviewer 1's comment 1 — Table 1 numbers changed without explanation; C-CoT row contradiction
   - reviewer 1's comment 2 — §4.1 vs Table 2 split inconsistency
   - reviewer 1's comment 3 — Newline-based step segmentation not answered
2. Response to reviewer 3
   - reviewer 3's overall comment — Grammatical errors throughout the manuscript
3. Summary of Round-2 Revisions

---

## Response to reviewer 1

### reviewer 1's overall comment

**reviewer's concern.** Thanks for the detailed revision. Most of my previous concerns have been addressed: the overclaimed step-level claims are properly softened, and the added t-SNE/AUC evidence and the new Section 5.4 analysis answer my main questions. A few points remain: (1) Table 1 numbers changed without explanation, and the response letter's claim that the original C-CoT row was unfilled contradicts the version reviewed; (2) §4.1 says the two backbones share the same held-out set, but Table 2 shows different test sizes; (3) the newline-based step-segmentation question was not answered. With these clarified, the reviewer would support acceptance.

**Our response.** We thank the reviewer for the positive overall assessment and for the three remaining points, which we address in turn below.

---

### reviewer 1's comment 1 — Table 1 numbers changed without explanation; C-CoT row contradiction

**reviewer's concern.** Almost all numbers in Table 1 changed from the previous version (e.g., Standard CoT on GSM8K/LLaMA went from 43.80 to 71.50), but this is never explained. Please state what was wrong in the original evaluation. Also, the response letter says the original C-CoT row was unfilled ("--"), but it had numbers in the version I reviewed, so please correct this.

**Our response.** We thank the reviewer for flagging both points and apologize for the lack of clarity in the first-round response letter.

**(a) Why the Table 1 numbers changed.** The changes are not a re-seed or a re-selection of favorable runs; they are the result of a full re-run of the main table undertaken in response to the first-round review. The reviewers' feedback prompted us to re-implement all baselines under a single unified pipeline (`newrun/`), which addressed four aspects raised during the first round:

- **(i) C-CoT characterization (Reviewer 2, Comment 1).** C-CoT (Chia et al., 2023) is a generation-time contrastive prompting method, not a confidence-based selector. We re-characterized it accordingly and re-ran it under the unified pipeline so that its results are directly comparable to the other methods.
- **(ii) Dataset alignment (Reviewer 2, Comment 2).** We aligned the dataset usage with the actual available data (CommonsenseQA: 500 questions, not 1,000) and standardized the split protocol across all settings.
- **(iii) ORM training refinement (Reviewer 2, Comment 3).** We refined the ORM implementation — removed 512-token truncation, added `pos_weight` class balancing, and switched to separate Q–A encoding — to strengthen the baseline.
- **(iv) Pipeline consistency (Reviewer 2, Comment 4).** The original main table and granularity ablation used different protocols; we unified them under a single pipeline so all results are directly comparable.

During the re-run, three dataset-specific configurations were updated:

1. **CommonsenseQA / Qwen3-8B.** Under the unified pipeline (regenerated path pool, `qid`-grouped split with zero cross-split overlap, 10 epochs), D-ProtoCoT moved from $87.71$ to $70.00$ and Self-Consistency from $86.98$ to $62.00$. CommonsenseQA / LLaMA-3.1-8B-Instruct was re-run at the same protocol for consistency.
2. **GSM8K / LLaMA-3.1-8B-Instruct.** Switched from a 42-question internal split to the official GSM8K test set (n=200) with per-backbone `qid`-grouped splits. Standard CoT moved from $43.80$ to $71.50$ (D-ProtoCoT from $80.95$ to $80.00$).
3. **StrategyQA.** The answer extraction procedure was updated to handle the model's [yes/no] wrapper; the recovered per-path correctness rate moved from $28.5\%$ to $46.8\%$. After re-extraction, D-ProtoCoT is $66.67$ and C-CoT is $76.81$.

All other baseline rows (Self-Consistency, ORM, Static-Prototype) shifted as a cumulative consequence of the regenerated path pool, the corrected answer extractor, and the per-backbone `qid`-grouped test splits under the unified pipeline. Rather than attribute each cell-level change to a single isolated cause, we report the cumulative effect of these corrections. The Qwen3-14B column and the Self-Certainty row are new additions in this revision (Reviewer 3, Comments 1 and 2) rather than re-runs of existing numbers.

**(b) Correction of the first-round response letter's C-CoT claim.** We thank the reviewer for identifying this inconsistency, and we sincerely apologize for the error in our first-round response. The original C-CoT row in Table 1 was not unfilled; it contained six values produced by an earlier, non-unified implementation (e.g., $64.40$ on CommonsenseQA/LLaMA-3.1-8B-Instruct and $87.00$ on GSM8K/LLaMA-3.1-8B-Instruct). The statement in the first-round response letter that this row was "previously unfilled (`--`)" was incorrect. The original C-CoT values were obtained under a non-unified pipeline, and the values now reported in Table 1 are produced by a re-run under the unified pipeline (`newrun/ccot_prompting.py`), which renders the C-CoT row directly comparable to the other rows.

**Change made.** No manuscript-level change is required for this comment beyond the values already reported in Table 1, which are the final re-run numbers. The record is corrected here in this response letter, which serves as the on-the-record correction of the prior claim about the C-CoT row.

---

### reviewer 1's comment 2 — §4.1 vs Table 2 split inconsistency

**reviewer's concern.** Section 4.1 says the two backbones share the same held-out question set, but Table 2 shows different test sizes (CSQA 50 vs 42, StrategyQA 28 vs 42). Which is correct? And why do the training set sizes differ by backbone?

**Our response.** We thank the reviewer for catching this inconsistency, and we confirm that the §4.1 sentence "the two backbone models share the same held-out question set" is inaccurate. We have corrected it in the revised manuscript.

The corrected statement reflects the actual pipeline: each backbone is processed independently, and the question-grouped 8:1:1 split is applied to that backbone's own sampled reasoning-path pool. The split sizes therefore differ across backbones, as Table 2 reports:

- **CommonsenseQA.** Qwen3-8B: 400 train / 50 val / 50 test (500 total); LLaMA-3.1-8B-Instruct: 336 train / 42 val / 42 test (420 total).
- **StrategyQA.** Qwen3-8B: 224 train / 28 val / 28 test (280 total); LLaMA-3.1-8B-Instruct: 336 train / 42 val / 42 test (420 total).
- **GSM8K.** Both backbones use the official test set (n=200); the train/val sizes differ because each backbone's available training-path pool differs (Qwen3-8B: 820 train / 91 val; LLaMA-3.1-8B-Instruct: 378 train / 42 val).

The per-backbone path pools differ in size because of intersection filters applied during the data-conversion step, not because of path-generation failures: every question in every pool has exactly K=10 sampled paths. The conversion scripts for the two backbones applied different filters:

- **CommonsenseQA / LLaMA-3.1-8B-Instruct.** The conversion step kept only questions whose wording also appeared in the Qwen3-8B / CommonsenseQA pool, yielding $420$ of the $500$ Qwen3-8B questions; the remaining $80$ Qwen3-8B questions did not appear in the LLaMA-3.1-8B-Instruct generation output and were dropped.
- **StrategyQA / Qwen3-8B.** The conversion step intersected the Qwen3-8B generation output with the official StrategyQA training set, yielding $280$ questions.
- **StrategyQA / LLaMA-3.1-8B-Instruct.** The conversion step kept the LLaMA-3.1-8B-Instruct generation output as-is (no intersection with the official training set), yielding $420$ questions.

The split sizes reported in Table 2 therefore reflect these per-backbone pool sizes; the 8:1:1 split is then applied independently to each pool. We acknowledge that the two backbones' conversion filters are not symmetric (one intersects with the other backbone's pool, the other with the official training set, and one is kept as-is); this is a data-pipeline inconsistency from the first-round revision that we are documenting honestly here. Critically, all splits obey the same grouping and zero-overlap protocol, so there is no cross-split question leakage within either backbone. This does not affect the within-backbone comparisons that support the main claims, since all methods (D-ProtoCoT and every baseline) on a given backbone share that backbone's test set. We note only that cross-backbone accuracy figures for CommonsenseQA and StrategyQA rest on different question subsets (GSM8K is unaffected, as both backbones use the same official n=200 test set); the paper's cross-backbone narrative concerns generalization across model families rather than point-to-point accuracy comparison.

Separately, within each backbone's pool, only questions with both correct and incorrect sampled paths contribute to contrastive training; the number of such trainable questions also varies across backbones (CommonsenseQA: $289$ for Qwen3-8B vs $200$ for LLaMA-3.1-8B-Instruct), reflecting differences in per-path correctness rates. This affects training-batch composition but not the split sizes reported in Table 2.

**Change made.** Reworded the §4.1 sentence — the original has been replaced as follows:

| # | Location | Before (original) | After (revised) |
|---|---|---|---|
| 1 | §4.1 (Datasets and Data Splits, held-out evaluation) | "...we report results on the question-grouped held-out split described above (a standard practice under this constraint); the two backbone models share the same held-out question set so that per-model results remain directly comparable." | "...we report results on the question-grouped held-out split described above (a standard practice under this constraint). **Each backbone model is processed independently. Reasoning paths are sampled separately per backbone, and the question-grouped 8:1:1 split is applied to that backbone's available question pool. Split sizes therefore differ across backbones (Table 2), reflecting differences in the per-backbone path pools. All splits obey the same grouping and zero-overlap protocol, so per-model results remain directly comparable under a consistent evaluation protocol.**" |

---

### reviewer 1's comment 3 — Newline-based step segmentation not answered

**reviewer's concern.** My question about newline-based step segmentation was not answered. Newline splitting can cut a step in half or merge steps, which directly affects the step-level training signal, so a quick comparison with explicit "Step k:" markers would be enough to show the method is robust to this choice.

**Our response.** We thank the reviewer for raising this point and apologize for not including the comparison in the previous revision. We have now run the comparison the reviewer requested on GSM8K / Qwen3-8B, the same setting used for the granularity ablation in §5.3.

**Setup.** We re-segmented the GSM8K / Qwen3-8B training paths using explicit `Step \d+:` markers as step boundaries — each step is the text from one `Step N:` marker to the next, with the final step extending to the end of the path (covering the `Final Answer:` line and any trailing reflection). The encoder was then retrained under otherwise-identical settings: same path pool, same `qid`-grouped 8:1:1 split, same test set (n=200 official GSM8K questions), same K = 10, 10 epochs, three seeds (42, 43, 44), learning rate $2 \times 10^{-5}$, temperature $\tau = 0.07$. The only difference from the run underlying Table 1 is the step-segmentation scheme.

**Granularity.** The two schemes produce substantially different step-level granularity on the same path pool: newline-based segmentation yields an average of $13.03$ non-empty steps per path, while `Step k:`-based segmentation yields $2.77$ steps per path — a $4.7\times$ difference, confirming that the reviewer's concern about the choice of delimiter has a material effect on the step-level training signal.

**Accuracy.** Across the three seeds, test-time selection accuracy is:

| seed | newline | `Step k:` |
|---|---|---|
| 42 | 86.50% | 82.00% |
| 43 | 82.00% | 81.50% |
| 44 | 80.00% | 81.00% |

Newline ranges from 80.00% to 86.50% and `Step k:` from 81.00% to 82.00%; the ranges overlap, and newline attains the higher accuracy in two of the three seeds. Neither delimiter consistently outperforms the other. We therefore conclude that the method is robust to the choice of step delimiter.

**Edge case.** Approximately $1.5\%$ of paths in the GSM8K / Qwen3-8B pool lack explicit `Step N:` markers; under the `Step k:` scheme these paths are treated as a single step spanning the whole path, which is the natural fallback. This affects a small fraction of training examples and does not change the conclusion.

**Why the comparison is on GSM8K only.** The `Step k:` scheme is only meaningful where paths actually contain explicit `Step N:` markers. We checked all six 8B path pools: only GSM8K paths reliably use such markers (LLaMA-3.1-8B-Instruct: 96.9% of paths contain them; Qwen3-8B: 69.7%). On CommonsenseQA and StrategyQA the sampled paths are free-form prose without `Step N:` markers — CommonsenseQA / Qwen3-8B: 99.9% of paths have none; StrategyQA / Qwen3-8B: 99.6%; CommonsenseQA / LLaMA-3.1-8B-Instruct: 87.8%; StrategyQA / LLaMA-3.1-8B-Instruct: 61.1%. On those datasets, `Step k:` segmentation would treat the vast majority of paths as a single step, degenerating to path-level pooling rather than step-level segmentation, so no meaningful delimiter comparison is possible there. Newline-based segmentation is used throughout the paper precisely because it is universally applicable: every reasoning path, across all backbones and datasets, contains newlines, whereas `Step N:` markers are a format-specific feature present only on GSM8K. GSM8K is therefore the only setting where the choice of delimiter can be tested, and there the method is robust to that choice.

**Change made.** Added a short paragraph and a table to §5.3 (Effect of Representation Granularity) reporting the newline-vs-`Step k:` segmentation comparison on GSM8K / Qwen3-8B. The full text and table added to the manuscript are:

> **[New]** (§5.3, Step-Delimiter Robustness): "The step-level segmentation used throughout this paper is based on newline delimiters, which are universally applicable across all benchmarks and backbones, whereas explicit `Step N:` markers appear only in GSM8K paths. A natural concern is that newline splitting may cut a step in half or merge steps, altering the step-level training signal. To check robustness to this choice, we re-segmented the GSM8K / Qwen3-8B training paths using explicit `Step N:` markers as boundaries and retrained the encoder under otherwise-identical settings (10 epochs, K=10, three seeds, n=200 official test questions). The two schemes yield substantially different granularity—13.03 vs. 2.77 steps per path on average. As Table 5 shows, the accuracy ranges under the two schemes overlap across the three seeds, and neither delimiter consistently outperforms the other. The method is therefore robust to the choice of step delimiter."
>
> **[New]** Table 5 (Step-delimiter robustness on GSM8K / Qwen3-8B):
>
> | Seed | Newline (%) | Step N: (%) |
> |---|---|---|
> | 42 | 86.50 | 82.00 |
> | 43 | 82.00 | 81.50 |
> | 44 | 80.00 | 81.00 |
> | Range | 80.00–86.50 | 81.00–82.00 |

No other parts of the pipeline change.

---

## Response to reviewer 3

### reviewer 3's overall comment — Grammatical errors throughout the manuscript

**reviewer's concern.** The authors responded to the previous comments and made the necessary corrections to the manuscript. Some of the descriptions in the manuscript contain certain grammatical errors. It is suggested that the authors recheck the entire manuscript and make the necessary revisions.

**Our response.** We thank the reviewer for the careful read and have performed a full grammatical pass over the entire manuscript, rechecking each section for punctuation, tense consistency, and phrasing.

**Change made.** Performed a full grammatical pass over the manuscript. The specific fixes are:

| # | Location | Issue | Before (original) | After (revised) |
|---|---|---|---|---|
| 1 | §2.2 (Reasoning Path Selection and Verification) | Redundant clause (repeats "propagates ... to every step" from the preceding sentence); the long sentence is also split into two | "As a result, D-ProtoCoT propagates the outcome-derived signal to every step, providing denser step-level supervision than outcome-level objectives at the same annotation cost---though, unlike PRMs, without step-level correctness labels." | "As a result, D-ProtoCoT **provides denser step-level supervision** than outcome-level objectives at the same annotation cost. **However, it does not require step-level annotation like PRMs.**" |
| 2 | §3.5 (Prototype-Based Alignment and Path Selection) | Punctuation around a display equation: the sentence and equation run together; a colon is added and the equation's trailing period changed to a comma | "The best reasoning path is selected by maximizing the alignment score $c^{*} = \arg\max_{c_i \in \mathcal{C}} a_i.$ and the final answer is extracted from $c^{*}$." | "The best reasoning path is selected by maximizing the alignment score**:** $c^{*} = \arg\max_{c_i \in \mathcal{C}} a_i$**,** and the final answer is extracted from $c^{*}$." |
| 3 | §4.2 (Backbone Models) | Dash usage: a single hyphen used as a dash is replaced with an em-dash, consistent with the rest of the manuscript | "two instruction-tuned LLMs **-** LLaMA-3.1-8B-Instruct and Qwen3-8B" | "two instruction-tuned LLMs**---**LLaMA-3.1-8B-Instruct and Qwen3-8B" |
| 4 | §5.5 (Impact of Backbone Model Strength) | Awkward phrasing: "in comparison to" → "compared to", extraneous comma removed, "modest" → "smaller" to convey the comparative meaning | "...the performance gap between self-consistency and D-ProtoCoT is modest**,** in comparison to weaker backbone models." | "...the performance gap between self-consistency and D-ProtoCoT is **smaller** compared to weaker backbone models." |
| 5 | Limitations (Second limitation) | Redundant phrasing: "the AUC-of-$0.78$ evidence ... is the empirical evidence" rewritten to remove the repeated noun and the awkward hyphenation | "...the AUC-of-$0.78$ evidence reported in Section 5 is the empirical evidence of what the representation does capture under this labeling scheme..." | "...the AUC of $0.78$ reported in Section 5 **empirically shows** what the representation does capture under this labeling scheme..." |

---

## Summary of Round-2 Revisions

We summarize the changes made in response to the round-2 feedback:

- **R1-C1 (Table 1 changes + C-CoT contradiction).** Full explanation of the unified-pipeline re-run and the three dataset-specific issues (CSQA 3→10 epoch, GSM8K test=42→200, StrategyQA extractor bug) that surfaced during it; on-the-record correction of the first-round response letter's claim that the original C-CoT row was unfilled. No manuscript-level change required beyond the final re-run numbers already in Table 1.
- **R1-C2 (§4.1 vs Table 2 split inconsistency).** Reworded the §4.1 "share the same held-out question set" sentence to a faithful per-backbone independent-split statement (each backbone has its own 8:1:1 split, sizes depend on the per-backbone path pool).
- **R1-C3 (Newline-based step segmentation).** Ran the newline vs. `Step k:` segmentation comparison on GSM8K/Qwen3-8B (10 epochs, K=10, three seeds, n=200 test). Newline: 80.00-86.50%; `Step k:`: 81.00-82.00%; ranges overlap, newline wins 2/3 seeds; neither consistently outperforms. Added a short paragraph ("Step-Delimiter Robustness") and a table (Table 5) to §5.3 reporting the comparison.
- **R3-C1 (Grammatical errors).** Performed a full grammatical pass over the manuscript (punctuation around display equations, dash usage, awkward phrasing, redundant clauses).

We believe the above revisions address the round-2 concerns, and we respectfully submit the revised manuscript for the editor's review.
