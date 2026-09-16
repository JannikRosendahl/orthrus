# ORTHRUS — detection, thresholding and ground truth

Extends §4 of [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md),
which established that detection is node-level over the edge-type objective. This note answers
the three questions the thesis section left open: **what unit is actually scored**, **every
threshold in the pipeline and which of them is live**, and **what the ground truth is and where
it comes from**.

Analysed at commit `0e11f7327c268eb40ae254e7ad446633fd5beec1` (2026-09-15).
Line references are to `src/` unless stated. Paper text: `resources/ORTHRUS.txt`.

**Provenance convention.** `[read]` = verified by reading the cited code during this analysis.
`[paper]` = claim from the paper. `[import-note]` = established in the companion import
analysis. `[measured]` = produced by a deterministic probe in
[`code/scratchpad/orthrus_analysis.ipynb`](../../../code/scratchpad/orthrus_analysis.ipynb).
`[unverified]` = inference, not confirmed by execution; never cite.

---

## 1. Detection granularity

### 1.1 The pipeline, end to end

`[read]`

| Stage | Produces | Location |
| --- | --- | --- |
| Testing | one CSV of per-edge losses per time window, per split, **per checkpoint** | `detection/orthrus_gnn_testing.py:12-88` |
| Node scoring | per-node score from incident edge losses | `detection/node_evaluation.py:12-63` |
| Thresholding | `y_hat` per node | `detection/evaluation_utils.py:23-31`, `:586-613` |
| Metrics | one stats dict per checkpoint | `provnet_utils.py:299-364` |
| Selection | the best checkpoint, by test MCC | `detection/evaluation.py:13-50` |

The scored unit is the **node**, not the window and not the edge, even though the model's
objective is per-edge. Windows survive only as the file granularity of the loss CSVs.

### 1.2 An edge's loss is charged to both of its endpoints

`[read]` `node_evaluation.py:33-35`:

```python
node_to_losses[srcnode].append(loss)
if cfg.detection.evaluation.node_evaluation.use_dst_node_loss:
    node_to_losses[dstnode].append(loss)
```

`use_dst_node_loss: True` is the shipped default (`config/orthrus.yml:71`), so a single
anomalous edge raises the score of the process **and** of the file or netflow it touched. A
node's score is then the max (or mean) over its collected losses
(`reduce_losses_to_score`, `evaluation_utils.py:33-39`), selected by the same
`threshold_method` string that selects the threshold — `max_val_loss` implies a max reduction.

This is what makes precision cheap on this benchmark: flagging the one busy process at the
centre of an attack also flags its files, and the ground truth lists those files as malicious
too.

### 1.3 Every node in the test split is a prediction

`[read]` `results` is keyed by every node id appearing in any test-window loss CSV
(`node_evaluation.py:48-58`), and `y_true = int(node_id in ground_truth_nids)` (`:53`). There is
no restriction to attack windows and no sampling of negatives, which is why the reported `TN` is
in the hundreds of thousands (`README.md:58-69`) `[import-note]`.

---

## 2. Thresholding

### 2.1 σ — the validation-loss threshold

`[read]` `get_threshold` (`evaluation_utils.py:23-31`) supports exactly two methods:

```python
if threshold_method == "max_val_loss":
    return calculate_threshold(val_tw_path)['max']
elif threshold_method == "mean_val_loss":
    return calculate_threshold(val_tw_path)['mean']
raise ValueError(...)
```

`calculate_threshold` (`:41-57`) pools **every edge loss in every validation window** and takes
the max, mean or 90th percentile. `max_val_loss` is the default (`config/orthrus.yml:70`), so
the operating point is the single largest validation edge loss — a max over millions of edges,
i.e. an extreme-value statistic with no stability guarantee. A `percentile_90` value is computed
and logged but is unreachable: the branch that would return it is commented out
(`evaluation_utils.py:29-30`). GRASP's observation that percentile thresholds change ORTHRUS's
results substantially (`resources/GRASP.txt:3602-3640`) `[paper]` therefore describes a variant
the shipped code cannot produce.

### 2.2 ⚠️ With `use_kmeans`, the threshold is computed, logged, and then discarded

`[read]` This is the finding that reframes every reported number.

`node_evaluation.py:55-61`:

```python
if use_kmeans: # in this mode, we add the label after
    results[node_id]["y_hat"] = 0
else:
    results[node_id]["y_hat"] = int(pred_score > thr)

if use_kmeans:
    results = compute_kmeans_labels(results, topk_K=cfg.detection.evaluation.node_evaluation.kmeans_top_K)
```

`use_kmeans: True` is the shipped default (`config/orthrus.yml:72`). In that mode every node
starts at `y_hat = 0` and `thr` is never consulted. `compute_kmeans_labels`
(`evaluation_utils.py:586-613`) then:

1. sorts all nodes by score,
2. keeps only the **last `kmeans_top_K = 20`** (`config/orthrus.yml:73`),
3. runs 2-means on those 20 scores,
4. sets `y_hat = 1` for the higher-centroid cluster only.

So **at most 20 nodes — in practice fewer — can be flagged in the entire test set**, regardless
of the threshold, the dataset size or how many nodes are genuinely anomalous. Recall is capped
structurally, and precision is inflated by construction.

`[import-note]` The authors' own results table confirms this operating: every `_ano` row has
`TP + FP ≤ 20`. See
[`metrics-splits-and-reported-results.md`](metrics-splits-and-reported-results.md) §1.4.

### 2.3 A supervised threshold exists but is unused

`[read]` `calculate_supervised_best_threshold` (`evaluation_utils.py:59-71`) fits a threshold
from the ROC curve **using the test labels**. It is dead code — nothing calls it — but it is
worth recording that it exists, because a reader auditing for label leakage will find it and
needs to know it is not on the live path.

---

## 3. Ground truth

### 3.1 What it is

`[read]` Per-attack CSVs under `Ground_Truth/darpa/`, listed per dataset in
`ground_truth_relative_path` and bounded in time by `attack_to_time_window`
(`config.py:214-222` CADETS_E3, `:173-181` THEIA_E3, `:259-266` CLEARSCOPE_E3). The directory is
a **nested git submodule** pointing at `ProvenanceAnalytics/ground-truth`, and it is not
populated by a plain clone — `git submodule update --init Ground_Truth/darpa` is required or the
labelling path cannot run at all.

Each row is `uuid,{'type': 'label'},index_id`. The labels are informative in themselves — CADETS
rows read `{'subject': 'None nginx'}`, which is the `"None <exec>"` subject featurisation
visible in the ground truth `[import-note]`.

`[measured]` The positive sets are small (probe P7):

| Corpus | attacks used | ground-truth rows | distinct uuids | resolved in PIDSMaker's DB |
| --- | ---: | ---: | ---: | ---: |
| E3-CADETS | 3 | 75 | 72 | 72 |
| E3-THEIA | 2 | 119 | 118 | 118 |
| E3-CLEARSCOPE | 1 | 41 | 41 | *(not imported)* |

Every distinct uuid resolves against PIDSMaker's import, so no ground-truth node is lost to the
import itself.

### 3.2 ⚠️ Ground-truth nodes that are never scored vanish from the confusion matrix

Two separate shrinkages sit between the ground-truth files and the reported `TP + FN`, and
neither is stated anywhere.

**First, de-duplication.** `get_ground_truth` (`labelling.py:6-21`) accumulates into a list and
returns `set(ground_truth_nids)` (`:21`) `[read]`. CADETS_E3's three attack files hold 75 rows
between them but only **72 distinct uuids** `[measured]` — three rows repeat across files.

**Second, and more consequential: only scored nodes enter the matrix.** `results` is keyed by
nodes that appear in a test-window loss CSV, and `y_true` is evaluated only for those
(`node_evaluation.py:48-58`) `[read]`. A ground-truth node that never appears in any scored test
window is therefore **absent from `results` entirely** — it is not counted as a false negative,
it is not counted at all.

The README reports `TP + FN = 68` on both CADETS rows (`README.md:60-61`), against 72 distinct
ground-truth uuids. `[unverified]` The most likely reading of the 4-node gap is exactly this
mechanism — those nodes do not occur in the test split's windows — but confirming it needs a run.
The mechanism itself is `[read]` and is not in doubt.

**Why it matters.** Reported recall is computed over *ground-truth nodes that happened to be
scored*, not over the ground truth. That is a denominator chosen by the pipeline rather than by
the evaluation protocol, and it moves in the optimistic direction: a positive the model never had
a chance to see is silently removed instead of counted against it. Any recall or MCC compared
against this system has to use the same denominator or say that it does not.

### 3.3 ⚠️ An unresolvable ground-truth uuid crashes the run

`[read]` `labelling.py:17`:

```python
node_id = uuid2nids[node_uuid]
```

A plain dict index, no `.get()` and no guard. `uuid2nids` is built from the three node tables of
the dataset's own database (`labelling.py:40-46`), so a ground-truth uuid that the importer
never stored raises `KeyError` and aborts evaluation rather than being counted as an
unreachable positive.

`[unverified]` This is very likely why one attack is commented out with the justification *"no
malicious nodes found in database"* (`config.py:261`): the file could not be used, not merely
that it would have scored zero. The same reading applies to the two THEIA phishing entries
(`:175-176`). The inference is consistent with the code but is not confirmed by running it.

### 3.4 Three attacks are excluded, and two of them have no file at all

`[read]` Commented-out entries, with the authors' stated reasons:

| Entry | Reason given | Line |
| --- | --- | --- |
| `E3-THEIA/node_Phishing_E_mail_Executable_Attachment.csv` | *"attack failed so we don't use it"* | `config.py:175` |
| `E3-THEIA/node_Phishing_E_mail_Link.csv` | *"attack only at network level, not system"* | `config.py:176` |
| `E3-CLEARSCOPE/node_clearscope_e3_firefox_0412.csv` | *"no malicious nodes found in database"* | `config.py:261` |

`[measured]` Neither THEIA phishing CSV exists in the pinned ground-truth submodule — the
`E3-THEIA` directory ships exactly the two files that are used. So those two attacks could not
have been scored even if uncommented, and the exclusion is not a curation decision that a
re-runner could reverse.

The positive set is therefore curated twice over: by which attacks DARPA's report describes, and
by which of those the authors could resolve against their own import.

### 3.5 Relation to the other ground truths in this corpus

- **KAIROS's** ground truth is four hardcoded 15-minute *window filenames*, not node uuids, and
  is unrelated in both granularity and provenance `[import-note]`. ORTHRUS's is node-level.
- **PIDSMaker** consumes exactly these CSVs — `ground_truth_version: orthrus` — so the ORTHRUS
  ground truth is the de-facto standard for every system evaluated in that framework, including
  VELOX and GRASP.
- **REAPr** is available on our server as node *and* edge labels
  (`reapr.{cadets,theia,fivedirections}_labels` / `*_edge_labels`), i.e. a strictly finer
  granularity than ORTHRUS's node lists. Choosing between them is a thesis decision, not an
  ORTHRUS finding; recorded here only so the comparison is known to be possible.

---

## 4. Open items

- Whether every ORTHRUS ground-truth uuid resolves against PIDSMaker's CADETS/THEIA imports. The
  notebook measures resolution (probe P7) against PIDSMaker's databases, which is a proxy: the
  standalone importer's node set differs `[import-note]`, so a mismatch there would not prove a
  mismatch in ORTHRUS's own run.
- The `attack_to_time_window` bounds are not used by the node-level path at all as far as this
  pass found; they feed `compute_tw_labels` (`evaluation_utils.py:277-332`) for per-window
  analysis and false-positive attribution. Whether any reported number depends on them was not
  established.
- The DEPIMPACT attack-reconstruction stage (`attack_reconstruction/`, ~820 LOC) is not analysed
  here. It is what separates the `_full` rows from `_ano`, so it matters for reading the results
  table, but it sits downstream of detection and does not bear on the memory question.
