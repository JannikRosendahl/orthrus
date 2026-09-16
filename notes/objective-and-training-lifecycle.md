# ORTHRUS — objective and training lifecycle

Companion to [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md)
(what the modules are) and [`graph-transformation-and-featurisation.md`](graph-transformation-and-featurisation.md)
(what reaches the tensor). This note answers **what is optimised**, **what the edge-type one-hot
inside the message means for the proxy task**, and **what state crosses the phase boundaries** —
the ORTHRUS counterpart of KAIROS's memory-lifecycle note, where the answer turns out to be
"almost none, and the little there is leaks".

Analysed at commit `0e11f7327c268eb40ae254e7ad446633fd5beec1` (2026-09-15).
Line references are to `src/` unless stated. Paper text: `resources/ORTHRUS.txt`.

**Provenance convention.** `[read]` = verified by reading the cited code during this analysis.
`[paper]` = claim from the paper. `[import-note]` = established in the companion import
analysis. `[unverified]` = inference or argument, not confirmed by execution; never cite.

---

## 1. The objective

### 1.1 It is a 10-way edge-type classification

`[read]` `EdgeTypeDecoder` is the only decoder in the repository (`decoders.py:4-41`). It
projects both endpoint embeddings, concatenates them, and maps the pair to one logit per edge
type:

```python
# decoders.py:35-42
def forward(self, h_src, h_dst, edge_type, inference, **kwargs):
    h = torch.cat([self.lin_src(h_src), self.lin_dst(h_dst)], dim=-1)
    h = self.lin_seq(h)

    edge_type_classes = edge_type.argmax(dim=1)
    loss = self.loss_fn(h, edge_type_classes, inference=inference)
    return loss
```

`num_edge_types = 10` for every dataset (`config.py:143-268`), and the class vocabulary is
`rel2id` (`config.py:622-644`) `[import-note]`. Unlike KAIROS — whose CADETS decoder has seven
classes against a paper that says nine — the configured width and the vocabulary agree here.

`predict_edge_contrastive` is referenced at `factory.py:39` and `model.py:25-27`, but
`decoder_factory` raises `ValueError` for anything other than `predict_edge_type`
(`factory.py:117`), so it is a dead branch.

### 1.2 The loss and the anomaly score are the same tensor

`[read]` The loss function is defined inline in the factory (`factory.py:99-101`):

```python
def cross_entropy(x, y, inference=False, **kwargs):
    reduction = "none" if inference else "mean"
    return F.cross_entropy(x, y, reduction=reduction)
```

so a training step minimises the mean cross-entropy over a batch's edges, while an inference
pass returns the *per-edge* cross-entropy unreduced. `Orthrus.forward` names this explicitly —
"Train mode: loss | Inference mode: edge scores" (`model.py:59`) — and sums the decoders'
outputs into one tensor (`:63-78`). Detection is therefore reconstruction error in the ordinary
sense: an edge is anomalous when its type is hard to predict from its endpoints.

### 1.3 The label is removed from the input it is predicted from

`[read]` `data_utils.py:134-138` drops the edge-type one-hot from `msg` exactly when the
objective is to predict it:

```python
# If we want to predict the edge type, we remove the edge type from the message
if "predict_edge_type" in cfg.detection.gnn_training.decoder.used_methods:
    msg = torch.cat([x_src, x_dst], dim=-1)
else:
    msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)
```

The guard is real and it is correct for the predicted edge. It is narrower than it looks,
though: `edge_type` is still supplied as an *edge feature* to the encoder's attention over
**neighbouring** edges (`encoders.py:70-76`, `edge_features: edge_type` at
`config/orthrus.yml:45`). So the type of the edge under prediction is hidden, while the types of
the surrounding edges are visible. That is defensible — it is context, not the label — but it
means the task rewards modelling the local type distribution, which is what makes a 10-way
type classification learnable at all.

### 1.4 Class imbalance is not handled

`[read]` `F.cross_entropy` is called with no `weight` argument and no sampling correction
(`factory.py:101`). The admitted 10 types are very unevenly distributed — a handful of
read/write operations account for most of the surviving stream on both E3 corpora
`[import-note]` — so the decoder's cheapest route to a low mean loss is to predict the majority
type. Since the *per-edge* loss is the anomaly score, this directly shapes which edges look
anomalous: rare-but-benign types carry a high score by construction. PIDSMaker exposes a
`balanced_loss` flag for precisely this; ORTHRUS has no equivalent.

### 1.5 Optimisation details

`[read]` Adam via `optimizer_factory` (`factory.py:138-142`), `num_epochs = 6`, `lr = 0.00001`,
`weight_decay = 0.00001`, `node_hid_dim = 128`, `node_out_dim = 64`, `dropout = 0.5`
(`config/orthrus.yml:37-41,51`). There is no early stopping, no learning-rate schedule and no
`patience` key at all — the training loop runs a fixed number of epochs
(`orthrus_gnn_training.py:50-95`).

⚠️ **A checkpoint is written after every epoch** (`orthrus_gnn_training.py:89-91`):

```python
if cfg._test_mode or epoch % 1 == 0:
    model_path = os.path.join(gnn_models_dir, f"model_epoch_{epoch}")
    save_model(model, model_path)
```

`epoch % 1 == 0` is true for every epoch, so all six are kept. That is not wasteful by accident —
it is what the evaluation stage consumes when it selects the reported epoch, and §2.4 of
[`metrics-splits-and-reported-results.md`](metrics-splits-and-reported-results.md) shows it does
so on the **test** set.

---

## 2. What crosses the phase boundaries

### 2.1 There is no memory to carry

`[read]` ORTHRUS holds no per-node state. `temporal.py:1-101` provides `IdentityMessage`,
`LastAggregator` and `LastNeighborLoader`; none of them retains a vector across edges, and no
`TGNMemory` or equivalent exists anywhere in `src/`. The only thing that persists between
batches is the neighbour loader's adjacency ring buffer — *which* neighbours a node last
interacted with, not *what* happened.

This is the structural difference from KAIROS, and it is why ORTHRUS's temporal signal is
ordering alone.

### 2.2 The neighbour loader is reset once per training epoch

`[read]` `orthrus_gnn_training.py:55-57`:

```python
# Before each epoch, we reset the memory
if isinstance(model.encoder, OrthrusEncoder):
    model.encoder.reset_state()
```

which forwards to `neighbor_loader.reset_state()` (`encoders.py:87-88`). Each epoch therefore
replays the training split from an empty adjacency. Note the comment says "memory" although
there is none — the reset clears the sampler.

### 2.3 ⚠️ At inference it is never reset, so validation state leaks into test

`[read]` `reset_state` is called in exactly one place in the whole repository — the training
loop above (grep over `src/` returns `temporal.py:35,99`, `encoders.py:87-88`,
`orthrus_gnn_training.py:57` and nothing else).

The testing stage builds one model per checkpoint (`orthrus_gnn_testing.py:108-112`) and then
scores both splits with it, in this order (`:115-117`):

```python
for graphs, split in [
    (val_data, "val"),
    (test_data, "test"),
]:
```

`OrthrusEncoder.forward` inserts every batch's edges into the loader as it goes
(`encoders.py:83`). Because nothing resets it between the two splits, **the adjacency the model
sees while scoring the test split still contains the entire validation split**, and the
validation split is not always earlier in time — on CADETS_E3 validation is day 2 while the test
days are 6, 11, 12, 13 (`config.py:211-212`) `[import-note]`.

The effect is bounded by `neighbor_size = 20` (`config/orthrus.yml:47`), so a busy node's
validation neighbours are quickly displaced by test ones; a node that appears in validation and
then rarely in test keeps them. It is a contamination of the encoder's input, not of the labels,
and it is small — but it is unreported, it runs in the direction of making test look more
familiar, and it is trivially avoidable with one `reset_state()` call.

### 2.4 Cold start

`[read]` A node seen for the first time has an empty neighbour list, so
`LastNeighborLoader.__call__` returns only the node itself and the encoder's attention has
nothing to attend over; the embedding reduces to the projection of its word2vec feature
(`encoders.py:62-78`). Since the word2vec feature is a deterministic function of the node's
label string, **two nodes with the same path and no history receive identical embeddings and
therefore identical predictions**. On CADETS, where every subject's label is `"None <exec>"`
`[import-note]`, that collapses a great many first-appearances onto a handful of vectors.

### 2.5 `--from_weights` short-circuits training entirely

`[read]` Every reproduction command in `README.md:80-95` passes `--from_weights`, and that flag
sets `num_epochs = 1` (`orthrus_gnn_training.py:49`), restricts the evaluated checkpoints to
`["model_epoch_1"]` (`orthrus_gnn_testing.py:101`) and overwrites the loaded parameters with the
shipped pickle (`:112`):

```python
if cfg._from_weights:
    model.load_state_dict(torch.load(os.path.join(cfg._from_weights_path, f"{cfg.dataset.name}.pkl")))
```

So the documented path reproduces the *published weights*, not a training run. Five pickles ship under `weights/` — `CADETS_E3`, `THEIA_E3`, `THEIA_E5`, `CLEARSCOPE_E3`,
`CLEARSCOPE_E5` — and ⚠️ **there is no `CADETS_E5.pkl`**, although `README.md:95` documents a
CADETS_E5 reproduction command that passes `--from_weights`. That command cannot run as written. Anyone re-training from scratch is on
an undocumented path, which is consistent with the README's own admission that the original
results could not be replicated exactly without `PYTHONHASHSEED=0`.

---

## 3. Consequences for the thesis comparison

- The objective is a **10-way classification whose per-edge loss doubles as the anomaly score**,
  with no imbalance correction. Any comparison of ORTHRUS's scores against a differently-shaped
  objective (this thesis's executable classification) has to account for the fact that its score
  distribution is dominated by edge-type frequency, not by behaviour.
- **Nothing persists across edges.** A TGN comparison that claims to add "memory" to ORTHRUS is
  adding the first per-node state the system has ever had — there is no weaker variant to
  degrade to, which makes ORTHRUS a clean zero-state baseline.
- The **validation-into-test sampler leak** (§2.3) and the **test-set checkpoint selection**
  (§1.5) are independent protocol defects that both flatter reported numbers. Neither is
  mentioned in the paper. They belong in the shortcomings list next to the non-chronological
  split, because they compound with it.

---

## 4. Open items

- Whether the sampler leak measurably moves the reported metrics. It needs a re-run with a
  `reset_state()` inserted between the splits, i.e. training ORTHRUS, which is out of scope for
  the related-work chapter and would need GPU time.
- Whether `patience: 3` is consumed anywhere outside the training loop. Not found in `src/`, but
  the config machinery resolves keys dynamically, so a negative here is weaker than a grep.
- The exact per-type frequency distribution over the admitted 10 types, which would quantify
  §1.4. Cheap on our own lossless import, but it belongs with the featurisation note rather
  than here.
