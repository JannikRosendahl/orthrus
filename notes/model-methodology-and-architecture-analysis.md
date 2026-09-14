# ORTHRUS — model, methodology & architecture analysis

Companion to [`orthrus-data-import-analysis.md`](orthrus-data-import-analysis.md),
which covers ingestion, featurisation and windowing. This note covers the **model**:
what the encoder actually computes, where time does and does not enter, and what the
published ablation does and does not show.

Analysed at commit `58f60bf192eaacf8bd71e640c7cd6e2867b12ce3` (2026-09-08).
Paper text: `resources/ORTHRUS.txt`.

**Provenance convention.** `[read]` = verified by reading the cited code in full during
this analysis. `[import-note]` = established in the companion import analysis.
`[paper]` = claim from the paper. `[unverified]` = plausible but not yet checked — do
not cite in the thesis until confirmed.

---

## 1. Headline: the encoder never reads the timestamp

`OrthrusEncoder.forward` takes `t` in its signature and **uses it nowhere in the body**
`[read]`:

```python
# src/encoders.py:50
def forward(self, edge_index, t, msg, x, full_data, inference=False, **kwargs):
    src, dst = edge_index
    x_src, x_dst = x
    ...
    n_id, edge_index, e_id = self.neighbor_loader(n_id)
    x_proj = self.src_linear(x_src[n_id]) + self.dst_linear(x_dst[n_id])
    h = x_proj
    # edge features: only edge_type and/or msg
    h = self.encoder(h, edge_index, edge_feats=edge_feats)
    ...
    self.neighbor_loader.insert(src, dst)
    return h_src, h_dst
```

Supporting facts, all `[read]`:

| Fact | Location |
| --- | --- |
| No `TimeEncoder`, no `dt`, no `last_update`, no `rel_t` anywhere in `src/` | repo-wide grep, zero hits |
| `time_encoding` occurs **exactly once** in the whole source — in a config comment | `src/config.py:90` |
| Selecting it raises `ValueError` | `src/factory.py:51-59` |
| `build_edge_feats` implements only `edge_type` and `msg` | `src/data_utils.py:150-158` |
| `temporal_dim` is only a hidden-layer width — nothing temporal | `src/encoders.py:46-47`, `factory.py:62` |
| No `TGNMemory` class exists; `src/temporal.py` has only `IdentityMessage`, `LastAggregator`, `LastNeighborLoader` | `src/temporal.py` |

The config comment is the trap:

```python
# src/config.py:90
"edge_features": str,  # ["edge_type", "msg", "time_encoding", "none"]
```

Reading the config alone suggests time encoding is available and merely switched off.
It is not implemented at all.

**Consequence.** The only temporal information reaching the model is the *ordering*
implicit in which edges currently occupy `LastNeighborLoader`'s last-N buffer. ORTHRUS
knows *that* an edge came before another; it never knows *by how much*.

**Why this matters for the thesis.** KAIROS, which ORTHRUS outperforms, *does* have a
genuine Fourier Δt encoder feeding edge attention
(`kairos/DARPA/CADETS_E3/model.py:17-29`, `rel_t = last_update[edge_index[0]] - t`)
`[read]`. So ORTHRUS wins while using **strictly less** temporal information — not just
no memory, but no time representation whatsoever. The literature reads this as "memory
is unnecessary". The defensible reading is broader and weaker: *on this benchmark,
under this ground truth and protocol, temporal information of any kind contributes very
little.*

---

## 2. What the encoder does compute

`[read]`, `src/encoders.py:22-88`:

1. Collect `n_id` = unique endpoints of the batch's edges.
2. Expand via `LastNeighborLoader(n_id)` → the last-N in-buffer neighbours per node,
   plus their edge ids `e_id`.
3. Project **static** node features: `x_proj = src_linear(x_src[n_id]) + dst_linear(x_dst[n_id])`.
   These are Word2vec features, fixed for the whole run — not state.
4. Assemble edge features from `edge_type` and/or `msg` only.
5. Run `GraphTransformer` — two `TransformerConv` layers `[read]`, `src/encoders.py:6-20`.
6. Return `h_src`, `h_dst`; insert the batch's edges into the neighbour buffer.

There is no per-node state that survives step 6 other than the neighbour buffer itself.
Node representations are recomputed from static features every forward pass.

Decoder: edge-type prediction (`config/orthrus.yml:84`, `predict_edge_type` /
`edge_mlp`) `[read]`.

---

## 3. The published ablation does not isolate memory

`[paper]` Table 6 (`resources/ORTHRUS.txt:1604-1650`) describes the four ablations. The
"Encoding" row is:

> With component: *ORTHRUS' encoder (§4.3)* — Without component: ***Kairos' TGN encoder***

and the discussion (`:1755-1765`):

> Substituting ORTHRUS' encoder with the TGN encoder from Kairos negatively affects
> performance on both datasets, as seen by the significant increase in false positives
> […] Additionally, using a memory vector for each node with TGN results in higher
> memory consumption.

This is a **whole-encoder swap**, not a memory toggle. Given §1, the swap
simultaneously changes at least:

- memory: absent → `TGNMemory` with GRU updates
- time representation: **absent → Fourier Δt encoder** (the big one, and unremarked)
- message function: none → `IdentityMessage` + `LastAggregator`
- propagation: `TransformerConv` over projected Word2vec → UniMP over memory state
- node featurisation: Word2vec → KAIROS's character-level hashes
- node keying: UUID → `sha256(label)` (see §5)

The paper's own framing (`:556-558`) — "our simple temporal sampling approach is
sufficient" — is therefore supported only in the weak sense that *ORTHRUS's whole
pipeline beats KAIROS's whole pipeline*. **No published experiment varies memory
alone.** That gap is this thesis's opening.

`[unverified]` The ablation harness itself has not been located in the code. Worth doing:
enumerate mechanically what the swap changes, to turn this argument into a diff.

---

## 4. Windowing, batching, splits

| Property | Value | Source |
| --- | --- | --- |
| Window size | 15.0 min | `config/orthrus.yml:5` `[import-note]` |
| Window cut | first 1024-edge boundary **after** 15 min elapsed → data-dependent length | `build_orthrus_graphs.py:107-146` `[import-note]` |
| Day tail | trailing partial window of each day **silently dropped** (no last-batch branch) | same |
| Neighbour sampling | temporal (`LastNeighborLoader`, last-N by insertion order) | `src/temporal.py:25-101` `[read]` |
| Batch order | chronological, non-shuffled | `src/data_utils.py:161-170` `[import-note]` |
| Node keying | UUID — one node per entity | `create_database/cadets_e3.py:115-118` `[import-note]` |

### Splits are not chronological

`src/config.py:143-260` `[import-note]`:

| Dataset | train | val | test |
| --- | --- | --- | --- |
| CADETS_E3 | 3, 4, 5, 7, 8, 9, 10 | 2 | **6**, 11, 12, 13 |
| CLEARSCOPE_E3 | 3, 4, 5, 7, 8, 9, 10 | 2 | 11, 12 |
| THEIA_E3 | 2, 3, 4, 5 | 9 | 10, 12, 13 |

For CADETS_E3 the test day **6** sits *inside* the training range 3–10. The model trains
on days 7–10, which occur *after* the day it is evaluated on. Defensible for a
per-window anomaly detector; **not** defensible as evidence of temporal generalisation,
and it should be flagged whenever ORTHRUS's numbers are quoted.

---

## 5. Edge fusion destroys the signal memory would use

`build_orthrus_graphs.py:176-203` `[import-note]`: within a window, edges are grouped by
`(src, dst)`, sorted by time, and **each run of consecutive identical operations
collapses to one edge**, carrying the run's first timestamp. There is no flag to
disable it.

A process reading a file 10,000 times in a window becomes **one edge**. Burst
structure, inter-arrival times and repetition counts — exactly the quantities a
stateful node memory is supposed to accumulate — are destroyed before any model sees
them.

This is the strongest available argument that the field's negative result on memory may
be a preprocessing artifact rather than a property of provenance data, and it is a
direct motivation for this thesis's lossless import.

---

## 6. Positioning on the thesis axes

| Axis | ORTHRUS |
| --- | --- |
| Node memory | **none** (no memory class in the repo) |
| Time representation | **none**; ordering only, via last-N neighbour buffer |
| Node keying | UUID (one node per entity) |
| Encoder | 2-layer `TransformerConv` over projected Word2vec features |
| Proxy task | edge-type prediction |
| Granularity | edge |
| Window | 15 min nominal, batch-quantised in practice |
| Split | day-based, **not chronological** on E3 |

---

## 7. Open items

- Locate the ablation harness; enumerate exactly what the "Kairos TGN encoder" swap
  changes (§3).
- Confirm whether `use_node_feats_in_gnn: False` paths are ever exercised, i.e. whether
  the encoder can run without the Word2vec projection at all.
- Compare against PIDSMaker's reimplementation of ORTHRUS — the configs are close but
  not identical, and if they diverge materially every VELOX comparison number is
  affected. See the PIDSMaker note.
