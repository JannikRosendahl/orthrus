# ORTHRUS — the published ablation, and why it cannot be reproduced

Extends §3 of [`model-methodology-and-architecture-analysis.md`](model-methodology-and-architecture-analysis.md),
which established that the "Encoding" ablation swaps a whole encoder and called the harness
`[unverified]` because it had not been located. This note closes that item. It answers
**what the paper actually claims about memory**, **what the artifact contains**, and **what a
controlled memory ablation would require**.

Analysed at commit `0e11f7327c268eb40ae254e7ad446633fd5beec1` (2026-09-15).
Line references are to `src/` unless stated. Paper text: `resources/ORTHRUS.txt`.

**Provenance convention.** `[read]` = verified by reading the cited code during this analysis.
`[paper]` = claim from the paper. `[import-note]` = established in the companion import
analysis. `[unverified]` = inference or argument, not confirmed by execution; never cite.

---

> **UPDATE 2026-09-16 — the harness was found, in VELOX's artifact.**
> `scripts/run_orthrus_ablation.sh` on the upstream `velox` branch
> (`upstream/velox` @ `54f687c`) reconstructs all four Table 6 ablations. It does not
> overturn this note's conclusion, and it settles §6's first open item. Its "Encoding"
> arm is
> `--detection.gnn_training.encoder.tgn.use_memory=True --detection.gnn_training.decoder.predict_edge_type.used_method=kairos`,
> so the swap **turns node memory on** *and* changes the edge-type decoder — two factors
> at once. §3 below reconstructs the swap by inference; that inference is now partly
> confirmed (memory is in it) and partly superseded (a decoder variant is in it too).
> The confound argument in §5 stands, and is now evidenced by the script rather than
> argued from absence. Featurisation is not touched by the arm, which answers §6's
> first bullet. Note the script is stale against that branch's schema —
> `src/config.py:232-236` defines no `used_method` under `predict_edge_type` — so it is
> a leftover, not a runnable reconstruction. Details:
> [`related-work/PIDSMaker/notes/velox/the-ablation-behind-the-claim.md`](../../PIDSMaker/notes/velox/the-ablation-behind-the-claim.md) §2a.
>
> The sentence below — "not in this repository" — remains true of *this* repository.

## 0. Answer in one paragraph

The ablation on which the field's "memory is unnecessary" reading rests is not in this
repository. `encoder_factory` constructs `GraphTransformer` + `OrthrusEncoder` unconditionally
and offers no alternative; there is no memory module, no encoder registry, and no configuration
key that could select "Kairos' TGN encoder" `[read]`. Worse for the received reading, the paper
does **not** claim that memory harms detection. It attributes the extra false positives to
*substituting the encoder*, and makes exactly one memory-specific claim — that a per-node memory
vector costs more **RAM** `[paper]`. So the literature's "ORTHRUS shows memory is unnecessary"
is stronger than what ORTHRUS says, and what ORTHRUS does say is not reproducible from its
artifact.

---

## 1. What the paper claims

`[paper]` Table 6 (`resources/ORTHRUS.txt:1605-1628`) describes four ablations as
component-level replacements. Two are relevant here:

| Component | With (✓) | Without (✗) |
| --- | --- | --- |
| Featurization | ORTHRUS' Word2vec embedding (§4.2) | Hierarchical hashing as in Kairos |
| Encoding | ORTHRUS' encoder (§4.3) | Kairos' TGN encoder |

Table 7 (`:1731-1747`) reports TP / FP / Precision / Memory per configuration on E3-THEIA and
E5-THEIA. The prose that interprets it (`:1755-1760`) `[paper]`:

> Substituting O RTHRUS' encoder with the TGN encoder from Kairos negatively affects performance
> on both datasets, as seen by the significant increase in false positives (this corroborates
> the lower detection quality of Kairos seen in Fig. 4). Additionally, using a memory vector for
> each node with TGN results in higher memory consumption.

Two distinct claims, and the distinction matters:

1. **Detection.** The FP increase is attributed to the *encoder substitution as a whole*. Memory
   is not named as its cause.
2. **Cost.** The only memory-specific claim is about **memory consumption in GB** — the `Memory`
   column of Table 7, which reads `4.23GB` for every row except the encoding ablation's
   `11.10GB` (`:1738-1742`).

So the paper supports "Kairos' encoder detects worse and a per-node memory costs RAM". It does
not, on its own text, support "node memory does not help detection". That inference is the
reader's, and §2 shows the artifact cannot settle it either.

**Featurization is a separate row.** Hierarchical hashing is ablated independently of the
encoder, so the encoding row does not additionally change the featuriser. That narrows the
confound relative to what §3 of the model note assumed, but does not remove it — see §3.

---

## 2. The harness is not in the artifact

`[read]` `encoder_factory` (`factory.py:43-91`) is the only construction path for an encoder.
It builds a `GraphTransformer` (`:66-74`) and wraps it in an `OrthrusEncoder` (`:81-92`), with
no branch, no registry lookup and no configuration key selecting an alternative:

```python
# factory.py:66-74
encoder = GraphTransformer(
    in_dim=in_dim, hid_dim=node_hid_dim, out_dim=node_out_dim,
    edge_dim=edge_dim or None, ...
)
...
encoder = OrthrusEncoder(encoder=encoder, neighbor_loader=neighbor_loader, ...)
return encoder
```

**Evidence for the negative** `[read]`:

- No memory module of any kind. `TGNMemory`, `IdentityMessage`-backed memory, GRU cell or
  `memory` tensor: none appear in `src/`. `temporal.py:1-101` defines `IdentityMessage`,
  `LastAggregator` and `LastNeighborLoader` — the neighbour sampler only; there is no state
  that survives an edge.
- No encoder alternative. `factory.py` has factories for model, encoder, decoder, optimizer and
  batch loader; only `decoder_factory` (`:93-119`) branches on a config value.
- No configuration key. `config/orthrus.yml` has no encoder-choice field; `encoder.used_methods`
  (the mechanism PIDSMaker uses for exactly this purpose) does not exist here.
- `time_encoding` appears only inside a *comment string* in the config schema
  (`config.py:90`), and requesting it raises `ValueError` (`factory.py:51-59`)
  `[import-note]`.
- The word "kairos" does not occur anywhere in `src/`, `config/` or `README.md`; the only
  match in the repository is the ground-truth filename
  `node_Firefox_Backdoor_Drakon_In_Memory.csv` (`config.py:174,180`), which is unrelated.

The ablations were therefore run with code that was not released. The artifact reproduces
ORTHRUS, not the comparison ORTHRUS reports.

---

## 3. What the swap moves, as far as it can be reconstructed

Because the harness is absent, this is a reconstruction from the two encoders as they exist in
their own repositories, not a diff of the code that produced Table 7. Treat the list as the set
of factors that *would* have moved together, not as a verified enumeration `[unverified]`.

| Factor | ORTHRUS's encoder | KAIROS's TGN encoder |
| --- | --- | --- |
| Node memory | none | `TGNMemory`, GRU-updated per edge |
| Time representation | **none** — `forward` receives `t` and never reads it (`encoders.py:51,54-86`) `[read]` | Fourier `TimeEncoder` on `rel_t = last_update − t` |
| Message function | no message passing into state | `IdentityMessage` over `[src_mem ‖ dst_mem ‖ t_enc ‖ msg]` |
| Embedding module | 2× `TransformerConv` over projected word2vec features | `TransformerConv` (UniMP) over memory + time encoding |
| Neighbour source | `LastNeighborLoader`, last-N by insertion order | same sampler, but reading memory |
| Node features into the GNN | word2vec projection (`encoders.py:64`) | node feature not used in the GNN by default |

At least four of these move at once. A result attributable to "memory" therefore requires
holding time encoding, message function and embedding module fixed — which no published ORTHRUS
experiment does.

**A latent defect in the one relevant lever that does exist.** `use_node_feats_in_gnn` looks
like it could isolate the node-feature factor, but it is broken `[read]`: it gates the
*construction* of the projections (`encoders.py:45-46`) while `forward` uses them
unconditionally (`:64`):

```python
# encoders.py:44-46
self.use_node_feats_in_gnn = use_node_feats_in_gnn
if self.use_node_feats_in_gnn:
    self.src_linear = nn.Linear(in_dim, temporal_dim)
    self.dst_linear = nn.Linear(in_dim, temporal_dim)

# encoders.py:64  -- runs whether or not the flag was set
x_proj = self.src_linear(x_src[n_id]) + self.dst_linear(x_dst[n_id])
```

Setting it to `False` raises `AttributeError`. The path is not merely unexercised; it cannot
run. This closes open item 2 of the model note.

---

## 4. Where an isolated memory ablation does exist

`[import-note]` PIDSMaker reimplements both systems behind one configuration surface, and there
memory is a single boolean: `use_memory: True` in `config/kairos.yml`, `False` in
`config/orthrus.yml`, with everything else in the encoder stack selected by
`encoder.used_methods`. That is the only place in the surveyed corpus where memory can be
toggled without moving anything else.

Two caveats before treating it as the missing experiment:

- PIDSMaker's `kairos.yml` and `orthrus.yml` also differ in featurisation
  (`hierarchical_hashing`, 16 dims vs. `word2vec`, 128) and in `node_label_features`, so the
  shipped configs are not a clean A/B either. Fixing those across arms is a precondition.
- PIDSMaker is not ORTHRUS: its importer repairs attribute-extraction defects that are present
  in the standalone artifact (see
  [`orthrus-data-import-analysis.md`](orthrus-data-import-analysis.md) §5.2). Numbers from it
  belong to PIDSMaker-ORTHRUS, not to the ORTHRUS paper.

---

## 5. Consequence for the thesis

The chapter's S1 ("no published experiment isolates node memory") is stronger than previously
recorded, and for a better reason. It is not only that the ablation is confounded — the
artifact does not contain it, and the paper never makes the detection claim the field cites it
for. The correct framing for the related-work chapter is:

- ORTHRUS reports that *Kairos' encoder* detects worse and that *a per-node memory* costs RAM.
- The step from there to "memory does not help detection" is an inference no published
  experiment supports.
- That gap is precisely what a controlled memory on/off ablation is positioned to fill, and it
  is what makes the question worth a thesis rather than a citation.

---

## 6. Open items

- Whether the unreleased harness reused ORTHRUS's featurisation or KAIROS's when swapping the
  encoder. Table 6 lists featurisation as a separate row, which suggests it was held fixed, but
  the code that would confirm it was not published. Not answerable from the artifact.
- The `Memory` column of Table 7 (4.23 GB vs 11.10 GB) is the only quantitative memory result in
  the paper; reproducing it needs the missing harness, so it cannot be checked here.
- Whether PIDSMaker's `use_memory` toggle, with featurisation held fixed across arms, reproduces
  the direction of Table 7. Cheap — four runs — and recorded in `PLAN_RW.md` as deferred to the
  Evaluation chapter rather than scheduled here.
