# ORTHRUS — metrics, splits and reported results

Fills the three paragraphs the thesis section left empty (metrics + imbalance, data & split,
reported results) and closes the last open item in §4 of
[`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md):
**which epoch's numbers are reported, and how they were selected**.

Analysed at commit `0e11f7327c268eb40ae254e7ad446633fd5beec1` (2026-09-15).
Line references are to `src/` unless stated. Paper text: `resources/ORTHRUS.txt`.

**Provenance convention.** `[read]` = verified by reading the cited code during this analysis.
`[paper]` = claim from the paper. `[import-note]` = established in the companion import
analysis. `[measured]` = produced by a deterministic probe in
[`code/scratchpad/orthrus_analysis.ipynb`](../../../code/scratchpad/orthrus_analysis.ipynb).
`[unverified]` = inference, not confirmed by execution; never cite.

---

## 1. Metrics

### 1.1 What is computed, and from what

`[read]` `classifier_evaluation` (`provnet_utils.py:299-364`) is the single place any reported
number is produced. It emits `precision`, `recall`, `fpr`, `fscore`, `accuracy`,
`balanced_acc`, `auc`, `ap`, `lr(+)`, `dor`, `mcc` and the raw `tp/fp/tn/fn`.

⚠️ **Two families of numbers under two different decision rules.** The confusion-matrix metrics
come from `y_test_pred` (`:302`), which under the shipped config is the k-means labelling
(§2.2 of [`detection-thresholding-and-ground-truth.md`](detection-thresholding-and-ground-truth.md)).
`auc` and `ap` come from the continuous `scores` (`:314`, `:317`):

```python
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
...
auc_val = roc_auc_score(y_test, scores)
ap      = ap_score(y_test, scores)
```

So precision/recall/MCC describe a rule that can flag at most `kmeans_top_K` nodes, while
AUC/AP describe the unthresholded ranking. The two cannot be read as properties of one detector.

**Contrast with KAIROS.** KAIROS calls `roc_auc_score` on *hard 0/1 predictions*, which
degenerates to balanced accuracy. ORTHRUS does not make that mistake — its `auc` is a genuine
AUC over continuous scores. The ORTHRUS number is the more meaningful of the two, and the two
papers' "AUC" columns are not comparable with each other.

### 1.2 ⚠️ ADP does not exist in this artifact

`[read]` There is no attack-detection-precision metric anywhere in `src/`. The obvious hook is
an empty stub (`provnet_utils.py:366-367`):

```python
def get_detected_attacks(cfg):
    cfg.dataset.attack_to_time_window
```

One expression statement, no return, no caller that uses a result. `grep -niE "\bADP\b|attack
discovery|attack detection"` over `src/`, `config/` and `README.md` returns nothing.

Whatever attack-level metric the paper reports, and whatever ADP figure downstream work quotes
for ORTHRUS, was computed outside this artifact. PIDSMaker does implement ADP and uses it for
`best_model_selection: best_adp`, so ADP numbers attributed to ORTHRUS in comparative tables are
PIDSMaker's, not the standalone system's. This matters for the thesis's metric chapter: ADP is
not reproducible from ORTHRUS as released.

### 1.3 The imbalance, quantified

`[measured]` Detection is node-level and every node in a scored window is a prediction, against
a positive set of a few dozen uuids (probe P8):

| Dataset | ground-truth nodes | nodes in the database | positive rate | max flaggable |
| --- | ---: | ---: | ---: | ---: |
| CADETS E3 | 75 rows / 72 distinct | 2,682,632 | 2.8 × 10⁻⁵ | 20 |
| THEIA E3 | 119 rows / 118 distinct | 1,258,362 | 9.5 × 10⁻⁵ | 20 |

The authors' own results table makes the same point from the other side: on CADETS_E3 it reports
`TN = 268,075` against `TP + FN = 68` (`README.md:60-61`), i.e. a positive rate near 2.5e-4 among
scored nodes. Accuracy and FPR are uninformative at this ratio, which is why MCC and precision
carry the comparison — and why the k-means cap (§1.4) dominates recall.

### 1.4 The reported table confirms the k-means cap

`[read]` `README.md:58-69` lists `_full` and `_ano` variants per dataset. `_ano` is detection
only; `_full` additionally runs DEPIMPACT attack reconstruction, which is enabled in the shipped
config (`config/orthrus.yml:75-82`).

Summing the authors' own positives for every `_ano` row:

| Row | TP | FP | TP + FP |
| --- | ---: | ---: | ---: |
| CADETS_E3_ano | 15 | 0 | **15** |
| THEIA_E3_ano | 2 | 0 | **2** |
| CADETS_E5_ano | 1 | 2 | **3** |
| THEIA_E5_ano | 2 | 0 | **2** |
| CLEARSCOPE_E3_ano | 1 | 5 | **6** |

Every one is `≤ 20 = kmeans_top_K` (`config/orthrus.yml:73`). The `_full` rows exceed it
(CADETS_E3_full flags 32) precisely because tracing adds nodes after the capped detection step.

This is the cap operating in the published numbers, not an inference about it. It also explains
the perfect precision in three of five `_ano` rows: a rule that emits at most a handful of the
highest-scoring nodes will rarely be wrong, and will rarely find much either — CADETS_E3_ano
recalls 15 of 68 ground-truth nodes, THEIA_E3_ano 2 of 118.

---

## 2. Data and splits

### 2.1 The E3 splits are not chronological

`[read]` Splits are lists of graph directories (`config.py:143-268`), applied after every day in
`start_end_day_range` has been built `[import-note]`:

| Dataset | train | val | test | unused |
| --- | --- | --- | --- | --- |
| CADETS_E3 | 3, 4, 5, 7, 8, 9, 10 | 2 | **6**, 11, 12, 13 | — |
| THEIA_E3 | 2, 3, 4, 5 | 9 | 10, 12, 13 | 11 |
| CLEARSCOPE_E3 | 3, 4, 5, 7, 8, 9, 10 | **2** | 11, 12 | 6, 13 |

(`config.py:210-213`, `:169-172`, `:255-258`.)

Only **CADETS_E3** is non-chronological in the sense that matters: it trains on days 3–10 and
**tests on day 6**, inside the training range. CLEARSCOPE_E3 is *not* the same case — its test
days 11–12 do follow training; what is odd there is that the **validation** day (2) precedes the
entire training range, as it also does for CADETS_E3. THEIA_E3 is chronological throughout.
(An earlier pass recorded "CLEARSCOPE_E3 does the same"; that is wrong, and the notebook asserts
the distinction so it cannot recur.) The CADETS_E3 arrangement is still
defensible for a per-window anomaly detector — no single graph is both trained and tested on —
but it does not support any claim about temporal generalisation, and it interacts badly with
the sampler leak described in
[`objective-and-training-lifecycle.md`](objective-and-training-lifecycle.md) §2.3.

### 2.2 `unused_files` silently drops whole days

`[read]` Days listed in `unused_files` are built and then never loaded, with no stated criterion
(`config.py:155`, `:172`, `:194`, `:213`, `:235`, `:258`). The cost is largest on E5 —
CADETS_E5 uses 5 of its 10 days, THEIA_E5 and CLEARSCOPE_E5 4 of 10.

`[measured]` On THEIA E3, the single unused day 11 is:

**7,339,776 filtered events — 18.39% of the whole dataset** — built and then never loaded.
No criterion for the exclusion is given anywhere in the repository.

### 2.3 Per-dataset hyperparameters differ from the shipped config

`[read]` `README.md:80-95` gives one reproduction command per dataset, and each overrides the
config. Collected:

| Dataset | Overrides |
| --- | --- |
| CADETS_E3 | `dropout=0.25`, `node_hid_dim=256`, `node_out_dim=256`, `lr=0.001`, `num_epochs=20`, `seed=4` |
| THEIA_E3 | `dropout=0.1`, `seed=2` |
| CLEARSCOPE_E3 | `time_window_size=1.0`, `dropout=0.1`, `seed=2` |
| CADETS_E5 | `node_out_dim=128`, `lr=0.0001`, `dropout=0.1`, `time_window_size=1.0` |

So the "single configuration, minimal tuning" reading of `config/orthrus.yml` does not hold:
the window size, the model width, the learning rate, the epoch count and the seed are all tuned
per dataset. Two consequences for citation: **"ORTHRUS uses 15-minute windows" is false for
CLEARSCOPE_E3 and CADETS_E5**, and any capacity comparison must use the per-dataset dimensions,
not the 128/64 in the config.

### 2.4 ⚠️ The reported epoch is selected on the test set

`[read]` This is the most consequential protocol finding in this note.

A checkpoint is saved every epoch (`orthrus_gnn_training.py:89-91`) `[import-note]`. The testing
stage scores **both** val and test with **every** checkpoint (`orthrus_gnn_testing.py:101`,
`:115-117`). Then `standard_evaluation` (`evaluation.py:13-50`) iterates the checkpoints and
keeps the best by MCC:

```python
# evaluation.py:19, :46-49
best_mcc, best_stats = -1e6, {}
for model_epoch_dir in listdir_sorted(test_losses_dir):
    stats = evaluation_fn(val_tw_path, test_tw_path, model_epoch_dir, cfg, ...)
    ...
    if stats["mcc"] > best_mcc:
        best_mcc = stats["mcc"]
        best_stats = stats
wandb.log(best_stats)
```

`stats` comes from `node_evaluation.main`, whose confusion matrix is computed over the **test**
split; the validation split contributes only the loss threshold
(`node_evaluation.py:16`). So the epoch that is reported is the one that scored best **on the
test set**. The same argmax is repeated to choose the checkpoint for tracing
(`attack_reconstruction/tracing.py:81-90`).

Validation is never used for model selection — the code carries the authors' own note that it
perhaps should be (`orthrus_gnn_testing.py:114`): *"TODO: we may want to move the validation set
into the training for early stopping"*.

**Why it matters.** With 6 default epochs (20 for CADETS_E3) and a positive set of a few dozen
nodes, selecting the argmax of test MCC over checkpoints is an optimistic estimator, and the
optimism grows with the number of checkpoints. It is a separate defect from the
non-chronological split and compounds with it. Neither is mentioned in the paper.

### 2.5 The documented path does not train

`[read]` Every command in `README.md:80-95` passes `--from_weights`, which collapses training to
one epoch and overwrites the parameters with a shipped pickle
(`orthrus_gnn_training.py:49`, `orthrus_gnn_testing.py:101`, `:112`) `[import-note]`. Five
pickles ship under `weights/`; **there is no `CADETS_E5.pkl`**, so the documented CADETS_E5
command cannot run as written.

---

## 3. Reported results

### 3.1 The repository's own table

`[read]` `README.md:58-69`, reproduced in §1.4 for the `_ano` rows. Columns are
TP / FP / TN / FN / Precision / MCC — **no recall column**, which is where the k-means cap would
be most visible. Reading recall off the table: CADETS_E3_ano 15/68 ≈ 0.22, THEIA_E3_ano 2/118 ≈
0.02.

### 3.2 The results are not exactly reproducible, by the authors' own statement

`[read]` `README.md:54-55`:

> The original results could not be exactly replicated due to a missing PYTHONHASHSEED affecting
> Gensim's Word2Vec, though the following experiments yield similar results in most cases.

Every documented command therefore sets `PYTHONHASHSEED=0`. The affected component is the
featuriser, so the discrepancy enters through node embeddings — the input to everything
downstream — rather than through a training nondeterminism that averaging could absorb.

### 3.3 Third-party re-evaluation contradicts the published numbers

`[paper]` GRASP reports that it could not reproduce ORTHRUS's detection results at all, with
defaults *or* with the authors' published per-dataset hyperparameters, and that ORTHRUS
"consistently fails to detect any attacks" under its re-run
(`resources/GRASP.txt:3602-3640`). It also notes that substituting the max threshold for a
99.9th/99th percentile "significantly reduces reproducibility". The PIDSMaker issue tracker
carries the same complaint (`resources/GRASP.txt:485`).

Taken with §2.4, the picture is coherent rather than mysterious: a reported number that is the
argmax of test MCC over checkpoints, produced by a featuriser sensitive to hash seeding, under a
decision rule that flags at most 20 nodes, is exactly the kind of number that does not survive
an independent re-run.

**For the thesis.** Any table that places ORTHRUS beside KAIROS or this thesis must say which
implementation produced each cell. The standalone artifact, PIDSMaker's `orthrus.yml` arm and
GRASP's re-evaluation are three different systems with three different data paths, and their
numbers are not interchangeable.

---

## 4. Open items

- Whether the argmax-over-checkpoints selection materially inflates the reported MCC. Bounding
  it needs the per-epoch stats, i.e. a full training run; out of scope here, and recorded as a
  candidate experiment rather than a claim.
- The paper's own results tables (Tables 4–5) were not transcribed here — this note covers the
  artifact's `README.md` table, which is what a reproduction attempt would meet first. The paper
  tables belong with the systematisation row in `notes/related-work-survey.md`.
- Whether PIDSMaker's ADP for its `orthrus` arm is computed over the same ground truth as the
  standalone system's node set. Needed before any ADP number is quoted in a comparison table.
