# ORTHRUS — graph transformation and featurisation

Companion to [`orthrus-data-import-analysis.md`](orthrus-data-import-analysis.md) (record
selection) and [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md)
(the model). This note answers the two questions the thesis section left open: **what
transformation sits between the stored records and the tensor**, and **what a node's feature
actually is** once it reaches the encoder.

Analysed at commit `0e11f7327c268eb40ae254e7ad446633fd5beec1` (2026-09-15).
Line references are to `src/` unless stated. Paper text: `resources/ORTHRUS.txt`.

**Provenance convention.** `[read]` = verified by reading the cited code during this analysis.
`[paper]` = claim from the paper. `[import-note]` = established in the companion import
analysis. `[measured]` = produced by a deterministic probe in
[`code/scratchpad/orthrus_analysis.ipynb`](../../../code/scratchpad/orthrus_analysis.ipynb).
`[unverified]` = inference, not confirmed by execution; never cite.

---

## 1. Graph transformation — the short answer

**There is no transformation stage.** The string `transformation` does not occur anywhere in
`src/` or `config/` `[read]` — ORTHRUS has no undirected conversion, no DAG projection and no
pseudo-graph rewrite. Those are options PIDSMaker added later, and reading ORTHRUS through
PIDSMaker's configuration surface invents a stage it does not have. The graph handed to the
model is directed and multi-edge.

That makes the *implicit* transformations the whole story, and there are four: direction
reversal at import, a construction-time type filter, batch-quantised windowing, and
unconditional edge fusion. Only the first is semantic; the other three destroy information.

---

## 2. What IS applied

### 2.1 Direction is semantic, and it is flipped at import

`[read]` Ten operation types are stored with `src` and `dst` swapped relative to the CDM record,
so that the edge always points the way information flows. The list is
`edge_reversed` (`config.py:600-613`) and it is applied inside the importer, not the graph
builder — every `create_database/*.py` passes `reverse=edge_reversed` to `store_event`
(`cadets_e3.py:273`, `theia_e3.py:246`, and identically in the other four).

Consequence: **the database already holds reversed direction.** Anything comparing ORTHRUS's
stored edges against a raw CDM stream — including our own lossless import — must reverse the
same ten types first, or the two disagree on direction for `READ`, `OPEN`, `EXECUTE`,
`RECVFROM`, `RECVMSG`, `MMAP`, `LSEEK`, `ACCEPT`, `READ_SOCKET_PARAMS` and
`CHECK_FILE_ATTRIBUTES`.

### 2.2 A second type filter runs at construction

`[read]` `include_edge_type = rel2id` (`build_orthrus_graphs.py:101`), tested per event at
`:133`. This is *in addition* to the 4-type blocklist applied at import
(`config.py:615-620`) `[import-note]`. The practical effect is that `EVENT_CLOSE`,
`EVENT_MMAP` and `EVENT_LSEEK` are present in Postgres and then silently dropped when the
graphs are built — the database is not the model's input, and counting rows in it overstates
what the model sees.

### 2.3 Windows close on batch boundaries, and each day's tail is discarded

`[read]` `gen_edge_fused_tw` (`build_orthrus_graphs.py:100-245`) reads one day at a time
ordered by `(timestamp_rec, event_uuid)` (`:123`), accumulates events in batches of
`BATCH = 1024` (`:140-141`), and closes a window when the batch's **last** event exceeds the
configured span (`:145-146`):

```python
window_size_in_sec = cfg.graph_construction.build_graphs.time_window_size * 60_000_000_000
if batch_edges[-1][-2] > start_time + window_size_in_sec:
```

Two things follow. A window is never shorter than `time_window_size` but its actual length is
data-dependent — a quiet period stretches it until 1024 more events accumulate. And the
variable is misnamed: `time_window_size` is multiplied by `60_000_000_000`, so the units are
nanoseconds, not seconds.

⚠️ There is **no final-batch flush**. The loop ends when the day's events run out, and whatever
is still in `temp_list` is cleared (`:240`) without being written. Every day therefore loses its
trailing partial window.

⚠️ **The window size is not 15 minutes everywhere.** `config/orthrus.yml:5` sets `15.0`, but the
repository's own reproduction commands override it to `1.0` for CLEARSCOPE_E3 and CADETS_E5
(`README.md:90`, `:95`). Any statement of the form "ORTHRUS uses 15-minute windows" is true of
the default config and false of two of the six documented runs.

### 2.4 Edge fusion collapses runs of identical operations

`[read]` Within a window, edges are grouped by `(src, dst)` (`build_orthrus_graphs.py:169-172`),
sorted by time (`:177`), and each maximal **run of consecutive identical operations** becomes a
single edge carrying the run's **first** timestamp (`:180-205`):

```python
for idx, item in enumerate(operation_list):
    if item == current_type:
        continue                       # still inside the run -- this event is dropped
    else:
        ...
        current_type = item
        current_start_index = idx
```

There is no flag. A process reading the same file 10,000 times in a row becomes one edge, and
the burst structure — precisely the signal a memory-based model would consume — is gone before
any model sees it.

*Determinism note:* `sorted(data, key=lambda x: x[0])` sorts on timestamp alone, which looks
like it could reorder ties arbitrarily. It cannot: Python's sort is stable and the list was
appended in the query's `(timestamp_rec, event_uuid)` order, so ties keep a deterministic
order. Checked because run boundaries depend on it.

### 2.5 The cost of §2.3 and §2.4, measured

`[measured]` Replaying ORTHRUS's exact windowing and fusion over our lossless import, with its
type allowlist and endpoint rule applied (probe P3/P4 in the notebook):

| Dataset | filtered events | in written windows | edges after fusion | windows written | fusion reduction | day-tail loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CADETS E3 | 18,699,410 | 18,205,696 | **6,454,730** | 1,000 | 64.55% | 2.64% |
| THEIA E3 | 39,922,077 | 39,720,960 | **17,000,142** | 521 | 57.20% | 0.50% |

So **34.52%** of CADETS's already-filtered stream and **42.58%** of THEIA's reaches the model as
edges. Against the unfiltered corpora (41,350,895 and 106,044,692 events) that is **15.6%** and
**16.0%** respectively — ORTHRUS models roughly one sixth of what was recorded, on both datasets.

The day-tail column is the cost of the missing final flush: 487 of CADETS's 18,266 batches and
201 of THEIA's 38,991 sit in a partial window that is never written.

The fusion figures supersede the whole-day approximation in
[`notes/event-type-filtering-extent.md`](../../../notes/event-type-filtering-extent.md), which
fused over UTC days rather than ORTHRUS's batch-quantised windows and therefore reported a
lower bound.

---

## 3. Featurisation

### 3.1 Two label pipelines, and only one reaches the model

`[read]` This is the single easiest thing to get wrong about ORTHRUS, because the config
strongly implies otherwise.

**Pipeline A — the graph node label.** `build_orthrus_graphs.py:10-81` assembles a label from
`construction.node_label_features` and, with `use_hashed_label: True`, stores its SHA-256 digest
on the networkx node.

**Pipeline B — the model's features.** `build_feature_word2vec.py:130` calls
`get_indexid2msg` (`provnet_utils.py:369-421`), which **re-reads the database from scratch** and
ignores pipeline A entirely:

| Node type | Label used for features | Controlled by |
| --- | --- | --- |
| subject | `path + ' ' + cmd` | `use_cmd: True` (`config/orthrus.yml:27`) |
| file | `path` | — |
| netflow | `remote_ip` only | `use_port: False` (`config/orthrus.yml:28`) |

So `use_hashed_label` never touches the model's input, and `node_label_features` does not
control the features. Reading `config/orthrus.yml` alone gives the wrong picture of what the
encoder sees.

Because netflow nodes are featurised on `remote_ip` alone, that one column *is* the node's
entire feature content — which is what makes the uuid-indexing defect in four of six importers
consequential rather than cosmetic (see
[`orthrus-data-import-analysis.md`](orthrus-data-import-analysis.md) §5.2).

### 3.2 The corpus is deduplicated by label string and is not split-aware

`[read]` `build_feature_word2vec.py:29` keys the corpus on the label itself:

```python
corpus[msg[1]] = tokens
```

so every node sharing a path or command contributes **one** training sentence, and the
vocabulary is much smaller than the node count. The corpus is built over every node in the
database with no split filter (`:10-30`), which makes the featurisation transductive: test-set
label strings are in the vocabulary at training time. That is unsupervised and label-free, so it
is not label leakage — but an inductive comparison needs the corpus restricted to train.

`[measured]` Corpus sizes and dedup ratios (probe P5):

| Dataset | nodes | distinct label strings | collapse |
| --- | ---: | ---: | ---: |
| CADETS E3 | 2,682,632 | **9,258** | 290× |
| THEIA E3 | 1,258,362 | 508,562 | 2.5× |
| CLEARSCOPE E3 | 369,101 | 76,983 | 4.8× |

The per-kind figures show where it comes from. On CADETS, 224,146 subjects carry **130** distinct
labels (1,724×), 155,322 netflow nodes carry 1,170 (133×), and 2,303,164 file nodes carry 7,958
(289×). On THEIA, files are nearly all distinct (793,899 → 506,737) while netflow collapses 921×
(186,100 → 202).

CADETS is the extreme case: the featuriser fits 128-dimensional vectors for a vocabulary of about
nine thousand strings, and every one of 2.68 M nodes draws its feature from that set. Two nodes
with the same path are indistinguishable to the encoder before any structure is considered.

### 3.3 Tokenisation, and two latent crashes

`[read]` Paths are split on `/` after collapsing backslash runs; netflow labels on `:` and `.`
(`provnet_utils.py:424-432`):

```python
def tokenize_subject(sentence: str):
    new_sentence = re.sub(r'\\+', '/', sentence)
    return word_tokenize(new_sentence.replace('/', ' / '))
```

Node vectors are a weighted mean of token vectors, weights declining linearly across the token
sequence, then L2-normalised (`embed_edges_feature_word2vec.py:43-49`). Two unguarded steps:

- `model.wv[word]` (`:45`) raises `KeyError` on a token absent from the vocabulary. With
  `min_count = 1` and a corpus covering the whole database this cannot fire on the shipped
  path — but it is why the corpus *must* stay unfiltered, so §3.2's transductivity is load-
  bearing rather than incidental.
- `sentence_vector / np.linalg.norm(sentence_vector)` (`:49`) has no epsilon, so a zero-norm
  vector yields `NaN` and propagates silently.

### 3.4 What the model actually receives per edge

`[read]` `gen_vectorized_graphs` (`embed_edges_feature_word2vec.py:66-105`) builds one `msg` row
per edge by concatenating, in order (`:87-93`):

```
[ src_type_onehot(3) | src_emb(128) | edge_type_onehot(10) | dst_type_onehot(3) | dst_emb(128) ]
```

= 272 dimensions. The `TemporalData` carries `src`, `dst`, `t` (int64 ns) and `msg`; there is no
per-edge label tensor — the "label" for the proxy task is recovered by slicing the edge-type
one-hot back out.

`data_utils.py:113-146` does that slicing and decides what becomes a node feature: `src_emb` /
`dst_emb`, optionally concatenated with the type one-hot
(`use_node_type_in_node_feats`, `:130`), and the edge-type one-hot is removed from the
message exactly when it is the prediction target (`:134-138`) — see
[`objective-and-training-lifecycle.md`](objective-and-training-lifecycle.md) §1.3.

### 3.5 Not featurised anywhere

`[read]` `[import-note]` Event `size`, `sequence`, `threadId`, `programPoint`,
`predicateObject2`, `properties`; subject pid / ppid / tgid / user / parent; file CDM subtype,
permissions, size; netflow `local_ip`, `local_port`, protocol, byte counts; node degree or age.
`t` is carried on the `TemporalData` and is never read by the encoder
(`encoders.py:51`) — so the timestamp survives all the way to the model and is then discarded.

---

## 4. Cross-dataset contrast

`[measured]` The subject label is `path + ' ' + cmd`, so what that string contains decides how
much the featuriser has to work with:

| Dataset | Substrate | subjects | with a path | distinct paths | distinct commands |
| --- | --- | ---: | ---: | ---: | ---: |
| CADETS E3 | lossless | 224,629 | **0** | 0 | 0 |
| CADETS E3 | PIDSMaker | 224,146 | **0** | 0 | 130 |
| THEIA E3 | lossless | 279,391 | 278,385 | 168 | 1,569 |
| THEIA E3 | PIDSMaker | 278,363 | 278,363 | 168 | 1,569 |
| CLEARSCOPE E3 | PIDSMaker | 12,469 | **0** | 0 | 44 |

CADETS carries no subject path **in the capture itself** — our lossless import and PIDSMaker's
independently agree on zero — so ORTHRUS's `"None <exec>"` is faithful, not a defect, and the
whole CADETS subject vocabulary is the executable name. THEIA carries real paths but only a few
hundred distinct ones. This bounds how much a path-based featuriser can distinguish on either
corpus, and it applies equally to this thesis, which inherits the same label source.

---

## 5. Open items

- Whether the `NaN` path in §3.3 ever fires in practice. It needs a run with instrumentation;
  cheap if ORTHRUS is ever executed, pointless to speculate about otherwise.
- The per-type frequency distribution over the admitted 10 types, which would quantify the class
  imbalance noted in [`objective-and-training-lifecycle.md`](objective-and-training-lifecycle.md)
  §1.4. Cheap on our lossless import; not run here because it does not change any claim in this
  note.
- E5 and CLEARSCOPE windowing were not replayed — only CADETS E3 and THEIA E3 are imported
  losslessly. The `time_window_size=1.0` override (§2.3) means CLEARSCOPE_E3 and CADETS_E5 would
  give materially different window counts, so the measured table must not be generalised to them.
