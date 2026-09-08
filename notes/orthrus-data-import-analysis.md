# ORTHRUS — data import & preprocessing analysis

Code pass over `related-work/orthrus` (fork of the ORTHRUS artifact by the KAIROS group).
Scope, as in the companion KAIROS and PIDSMaker notes: **what enters the model** — how
records are selected, what becomes a node or an edge, which attributes survive, and every
filtering or transformation decision that can move downstream results.

Provenance format: `path:line`. Repo HEAD at analysis time: `e7f25df`.

## Sources

| Stage | File |
| --- | --- |
| Schema | `postgres/init-create-databases.sh`, `settings/scripts/create_database.sh` |
| Import dispatch | `src/create_database.py` |
| Per-dataset import | `src/create_database/{cadets,theia,clearscope}_{e3,e5}.py` |
| Edge/node vocabularies, dataset config | `src/config.py` |
| Graph construction | `src/graph_construction/build_orthrus_graphs.py` |
| Featurization | `src/edge_featurization/build_feature_word2vec.py`, `embed_edges_feature_word2vec.py` |
| Tokenizers, time, hashing, label read-back | `src/provnet_utils.py` |
| Run settings | `config/orthrus.yml` |

Coverage: **six DARPA TC datasets only** — CADETS / THEIA / CLEARSCOPE × E3 / E5
(`src/create_database.py:5-18`). There is no OpTC path anywhere in the repo.

---

## 1. The shared pipeline

Four stages.

1. **Schema** — `postgres/init-create-databases.sh` creates one database per dataset with
   four tables (§1.2).
2. **Import** — one hand-written module per dataset → Postgres.
3. **Construction** — `build_orthrus_graphs.py` → one `networkx.MultiDiGraph` per time
   window, under `graph_<day>/<tw_interval>`.
4. **Featurization** — word2vec over node labels, then one `TemporalData` per window.

Input files are discovered by glob, not by a hardcoded list:

```python
# src/provnet_utils.py:484-486
def get_all_filelist(filepath):
    files = glob.glob(f"{filepath}/*json*")
    return files
```

⇒ whatever `*json*` sits in `cfg.dataset.raw_dir` is imported. There is no record of
which files a given database was built from.

### 1.1 Record selection — every gate, verbatim

Every module reads the raw CDM JSON **line by line as text** and selects records with a
substring test, then extracts fields with positional regexes. No JSON parsing anywhere.

| Dataset | Target | Exact condition | Where |
| --- | --- | --- | --- |
| CADETS E3 | subject/file uuid pre-pass | `if "com.bbn.tc.schema.avro.cdm18.Subject" in line:` … `elif "com.bbn.tc.schema.avro.cdm18.FileObject" in line:` | `cadets_e3.py:16-25` |
| CADETS E3 | netflow | `if "NetFlowObject" in line:` **and** `if "FILE_OBJECT_UNIX_SOCKET" in line:` | `cadets_e3.py:36`, `:55` |
| CADETS E3 | subject | `if "Event" in line:` | `cadets_e3.py:97` |
| CADETS E3 | file | `if "Event" in line:` | `cadets_e3.py:134` |
| CADETS E3 | event | `if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line:` + `relation_type not in exclude_edge_type` | `cadets_e3.py:211-213` |
| CADETS E5 | as CADETS E3 with `cdm20` | `"schema.avro.cdm20.Subject"` / `"schema.avro.cdm20.FileObject"` / `"avro.cdm20.NetFlowObject"` / `"schema.avro.cdm20.Event"` | `cadets_e5.py:17-21`, `:30`, `:98`, `:134` |
| THEIA E3 | netflow / subject / file | `"NetFlowObject" in line` / `"schema.avro.cdm18.Subject" in line` / `"avro.cdm18.FileObject" in line` | `theia_e3.py:17`, `:70`, `:118` |
| CLEARSCOPE E3 | same three | same substrings | `clearscope_e3.py:17`, `:70`, `:109` |
| THEIA E5 / CLEARSCOPE E5 | same three | `cdm20` variants | `theia_e5.py`, `clearscope_e5.py` |
| all | event | `'{"datum":{"com.bbn.tc.schema.avro.cdmNN.Event"' in line` + `exclude_edge_type` | per module |

Observations:

- **CADETS E3/E5 take subject and file identity from `Event` records, not from entity
  records.** The gate `if "Event" in line:` (`cadets_e3.py:97`) is a bare substring test
  that also matches any line where `Event` appears in a path or argument.
- The remaining four datasets read `Subject` / `FileObject` records directly.
- Most gates are **unanchored**: `"NetFlowObject" in line` matches the type name anywhere
  in the record, not only in the datum position. Only the event gates are anchored on
  `{"datum":{…}`.
- `relation_type = re.findall('"type":"(.*?)"', line)[0]` runs **before** the exclusion
  check and takes the **first** `"type"` key in the line. It is unguarded — a matching
  line with no `"type"` raises `IndexError`.

### 1.2 CDM record types → the four tables

```sql
-- postgres/init-create-databases.sh:18-60
CREATE TABLE event_table (
    src_node VARCHAR, src_index_id VARCHAR, operation VARCHAR,
    dst_node VARCHAR, dst_index_id VARCHAR, event_uuid VARCHAR NOT NULL,
    timestamp_rec BIGINT, _id SERIAL PRIMARY KEY);
CREATE TABLE file_node_table    (node_uuid, hash_id, path, index_id);
CREATE TABLE netflow_node_table (node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port, index_id);
CREATE TABLE subject_node_table (node_uuid, hash_id, path, cmd, index_id);
```

| Table | CDM record type | Notes |
| --- | --- | --- |
| `subject_node_table` | `Subject` | stores **both** `path` and `cmd` |
| `file_node_table` | `FileObject` | one `path` column |
| `netflow_node_table` | `NetFlowObject` **and, in CADETS E3/E5, `FILE_OBJECT_UNIX_SOCKET`** | see below |
| `event_table` | `Event` | carries `event_uuid`, so edges are individually addressable |

**Unix sockets are typed as netflow in CADETS**, with all four address fields `None`:

```python
# cadets_e3.py:55-67
if "FILE_OBJECT_UNIX_SOCKET" in line:
    res = re.findall('FileObject":{"uuid":"(.*?)"', line)
    nodeid = res[0]
    srcaddr = None; srcport = None; dstaddr = None; dstport = None
    nodeproperty = [srcaddr, srcport, dstaddr, dstport]
```

and they are simultaneously excluded from the file table (`cadets_e3.py:21-22`,
`continue`). THEIA and CLEARSCOPE do not do this — there, unix sockets stay `FileObject`s
or are dropped. So the meaning of "netflow node" differs by dataset.

- `Subject.type` and `FileObject.type` are otherwise never read: `SUBJECT_PROCESS` /
  `_THREAD` / `_UNIT` and `FILE_OBJECT_FILE` / `_DIR` / `_NAMED_PIPE` all collapse into a
  3-valued node typing (`config.py:645-652`, `ntype2id`).
- **CDM record types never imported**: `Principal`, `MemoryObject`, `SrcSinkObject`,
  `UnnamedPipeObject`, `IpcObject`, `RegistryKeyObject`, `PacketSocketObject`,
  `ProvenanceTagNode`, `TagRunLengthTuple`, `Value`, `CryptographicHash`,
  `UnitDependency`, `Host`, `TimeMarker`, `StartMarker`/`EndMarker`.

### 1.3 Node identity

```python
# cadets_e3.py:115-118 (and identically in every module)
datalist.append([i] + [stringtomd5(i)] + subject_obj2hash[i] + [index_id])
#                       ^^^^^^^^^^^^^^ i is the UUID
subject_uuid2hash[i] = stringtomd5(i)
```

- **All node types are keyed on `sha256(uuid)`** — one node per entity. `stringtomd5` is
  named for MD5 but calls `hashlib.sha256` (`provnet_utils.py:45`).
- `index_id` is assigned incrementally **during import**, across the three tables in the
  order **netflow → subject → file** (`cadets_e3.py:251-265`). It is the integer the
  model uses as node id.
- The `len(i) != 64` guard on every insert loop selects uuid keys out of the
  bidirectional `{uuid: [hash, …], hash: uuid}` dicts.

### 1.4 Field extraction, per dataset

Subject label fields — **the most divergent part of the codebase**:

| Dataset | `path` | `cmd` | Source |
| --- | --- | --- | --- |
| CADETS E3 | **always `None`** | `"exec"` from Event lines | `cadets_e3.py:97-107` |
| CADETS E5 | **always `None`** | `"exec"` from Event lines | `cadets_e5.py` |
| THEIA E3 | `'"properties":{(.*?)"path":"(.*?)","ppid"'` | `',"cmdLine":{"string":"(.*?)"},'` | `theia_e3.py:70-93` |
| THEIA E5 | group 2 of the Subject regex | `',"cmdLine":{"string":"(.*?)"},'` | `theia_e5.py` |
| CLEARSCOPE E3 | **hardcoded `None`** | ⚠️ `subject_cmd[0][0]` | `clearscope_e3.py:74-81` |
| CLEARSCOPE E5 | **hardcoded `None`** | `subject_cmd[0]` | `clearscope_e5.py` |

In CADETS the `path` column is filled from a pre-pass dict that is **populated with
`None` and never updated**:

```python
# cadets_e3.py:16-19
subject_uuid2path[match_ans] = None      # registered, never filled
...
# cadets_e3.py:100
subject_obj2hash[subject_uuid[0][0]] = [subject_uuid2path[subject_uuid[0][0]], subject_uuid[0][-1]]
#                                        ^^^^ always None                       ^^^^ the exec name
```

⚠️ **CLEARSCOPE E3 stores the first character of the command line**, not the command line:

```python
# clearscope_e3.py:74-81
subject_cmd = re.findall(',"cmdLine":{"string":"(.*?)"},', line)   # -> list[str]
if len(subject_cmd) == 0:
    node_cmd = None
else:
    node_cmd = subject_cmd[0][0]        # first char of the first match
```

Every other module uses `subject_cmd[0]`.

File label:

| Dataset | Source | Extra gate |
| --- | --- | --- |
| CADETS E3 | `'"predicateObjectPath":{"string":"(.*?)"}'` on Event lines | `'"predicateObjectPath":null,' not in line and '<unknown>' not in line` (`cadets_e3.py:139`) |
| CADETS E5 | same, on Event lines | **no `<unknown>` / null gate** |
| THEIA E3/E5 | `"filename"` on FileObject records | — |
| CLEARSCOPE E3/E5 | `"path"` on FileObject records | — |

Netflow: ⚠️ **THEIA E3/E5 and CLEARSCOPE E3/E5 store two characters of the node UUID as
the remote address and port.**

```python
# theia_e3.py:18-36 (identical in clearscope_e3, theia_e5, clearscope_e5)
res = re.findall('NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),', line)[0]
nodeid  = res[0]
srcaddr = res[2]
srcport = res[3]
remote = re.findall('"remoteAddress":"(.*?)","remotePort":(.*?),', line)
if len(remote) > 0:
    dstaddr = res[0][0]      # <-- res[0] is the UUID string; [0] is its first character
    dstport = res[0][1]      # <-- its second character
else:
    dstaddr = None
    dstport = None
```

`remote` is matched and its length tested, but its captures are never used — `res[0][0]`
and `res[0][1]` index into the uuid. CADETS E3/E5 do not have this bug; they read
`res[4]` / `res[5]` from a single regex covering all four fields
(`cadets_e3.py:37-48`). Since ORTHRUS features netflow nodes on **`remote_ip`**
(§1.6), the netflow feature for four of six datasets is a single hex character.

Events — five regexes, identical in every module:

- `'"subject":{"com.bbn.tc.schema.avro.cdmNN.UUID":"(.*?)"'`
- `'"predicateObject":{"com.bbn.tc.schema.avro.cdmNN.UUID":"(.*?)"'`
- `'"type":"(.*?)"'`
- `'"timestampNanos":(.*?),'`
- `'{"datum":{"com.bbn.tc.schema.avro.cdmNN.Event":{"uuid":"(.*?)",'` → `event_uuid`

Discarded from every event: `size`, `sequence`, `threadId`, `programPoint`,
`predicateObject2`, `properties`, `hostId`, `location`, `name`, `parameters`.

### 1.5 Edge construction and the edge-type vocabularies

All three lists are global and shared by all six datasets (`src/config.py:600-644`):

- **`exclude_edge_type`** — blacklist applied at **import**, 4 entries, each with a
  justification comment: `EVENT_FCNTL`, `EVENT_OTHER`, `EVENT_ADD_OBJECT_ATTRIBUTE`,
  `EVENT_FLOWS_TO`.
- **`edge_reversed`** — 10 entries: `EVENT_EXECUTE`, `EVENT_LSEEK`, `EVENT_MMAP`,
  `EVENT_OPEN`, `EVENT_ACCEPT`, `EVENT_READ`, `EVENT_RECVFROM`, `EVENT_RECVMSG`,
  `EVENT_READ_SOCKET_PARAMS`, `EVENT_CHECK_FILE_ATTRIBUTES`.
- **`rel2id`** — 10 types, both the one-hot vocabulary and the construction filter
  (`build_orthrus_graphs.py:101`, `include_edge_type = rel2id`): `CONNECT, EXECUTE, OPEN,
  READ, RECVFROM, RECVMSG, SENDMSG, SENDTO, WRITE, CLONE`.

Endpoint rule (`cadets_e3.py:214-217`):

```python
if subject_uuid[0] in subject_uuid2hash and (predicateObject_uuid[0] in subject_uuid2hash or
                                             predicateObject_uuid[0] in file_uuid2hash or
                                             predicateObject_uuid[0] in net_uuid2hash):
```

- Src must be a subject; dst may be a **subject**, file or netflow ⇒ **subject→subject
  edges are kept for every dataset**, so the process tree is present.
- Resolution is an `if/elif/else` chain, file → netflow → subject, so no silent overwrite.
- Direction is swapped when `relation_type in edge_reversed`.
- `EVENT_CLOSE` and `EVENT_MMAP` are stored in Postgres but absent from `rel2id`, so they
  are dropped at construction.

### 1.6 Features

ORTHRUS builds **two independent label pipelines**. This is easy to miss and matters.

**Pipeline A — the graph node `label` attribute** (`build_orthrus_graphs.py:10-81`),
driven by config:

```python
features_used = []
for label_used in node_label_features['subject']:
    features_used.append(attrs[label_used])
label_str = ' '.join(features_used)
if use_hashed_label:
    nodeid2msg[hash_id] = ['subject', stringtomd5(label_str)]
```

with `config/orthrus.yml:1-10`:

```yaml
    use_hashed_label: True
    node_label_features:
      subject: type, path, cmd_line
      file: type, path
      netflow: type, remote_ip, remote_port
```

⇒ the label stored on every graph node is a **SHA-256 hex digest**. Available netflow
attributes include `local_ip` / `local_port` (`build_orthrus_graphs.py:25-30`) but the
config selects only `remote_*`.

**Pipeline B — the features actually fed to the model** re-reads the database and ignores
Pipeline A entirely (`build_feature_word2vec.py:130`, `provnet_utils.py:369-421`):

```python
indexid2msg = get_indexid2msg(cur, use_cmd=use_cmd, use_port=use_port)
```

| Node type | Label used for features | Controlled by |
| --- | --- | --- |
| subject | `path + ' ' + cmd` | `use_cmd: True` |
| file | `path` | — |
| netflow | `remote_ip` (no port) | `use_port: False` |

So `use_hashed_label: True` never touches the model's input; it only affects the label
carried on graph nodes for downstream inspection/tracing.

**Tokenization** (`provnet_utils.py:424-432`):

```python
def tokenize_subject(sentence):  # and tokenize_file, identically
    new_sentence = re.sub(r'\\+', '/', sentence)
    return word_tokenize(new_sentence.replace('/', ' / '))

def tokenize_netflow(sentence):
    return word_tokenize(sentence.replace(':', ' : ').replace('.', ' . '))
```

**Embedding** (`build_feature_word2vec.py`, `embed_edges_feature_word2vec.py`):

- Corpus = tokenized labels, **deduplicated by label string** (`corpus[msg[1]] = tokens`,
  `build_feature_word2vec.py:29`), over **every node in the database** — no split
  filtering.
- word2vec: `emb_dim=128`, skip-gram, `window=5`, `min_count=1`, `negative=5`,
  `epochs=50`, seeded from `cfg._seed`.
- Node vector = weighted mean of token vectors, weights linearly declining across the
  token sequence (`cal_word_weight`, `decline_rate=30`), then L2-normalized:

```python
# embed_edges_feature_word2vec.py:44-49
word_vectors = [model.wv[word] for word in tokens]          # no OOV guard
weighted_vectors = [w * v for w, v in zip(weight_list, word_vectors)]
sentence_vector = np.mean(weighted_vectors, axis=0)
normalized_vector = sentence_vector / np.linalg.norm(sentence_vector)   # no epsilon
```

⚠️ `model.wv[word]` raises `KeyError` on an unseen token, and the division has no
epsilon, so a zero-norm vector yields `NaN`.

**`msg` layout** (`embed_edges_feature_word2vec.py:87-93`):

```
msg = [ src_type_onehot(3) | src_emb(128) | edge_type_onehot(10) | dst_type_onehot(3) | dst_emb(128) ]
```

= 272 dims. The `TemporalData` handed to training carries `src`, `dst`, `t` (int64 ns) and
`msg`; there is no per-edge label tensor.

**Feature selection at training time** (`src/data_utils.py:120-146`) slices `msg` back
apart. Node features are `src_emb`, optionally concatenated with `src_type`
(`use_node_type_in_node_feats: True` in `config/orthrus.yml`), and the edge-type one-hot
is **dropped from the message when the objective is to predict it**:

```python
# src/data_utils.py:134-138
# If we want to predict the edge type, we remove the edge type from the message
if "predict_edge_type" in cfg.detection.gnn_training.decoder.used_methods:
    msg = torch.cat([x_src, x_dst], dim=-1)
else:
    msg = torch.cat([x_src, x_dst, fields["edge_type"]], dim=-1)
```

⇒ the label the decoder predicts is not present in the input it predicts from.

**Not featurized anywhere**: event `size` / `sequence` / `threadId` / `predicateObject2`
/ `properties`; subject pid / ppid / tgid / user / parent; file CDM subtype / permissions
/ size; netflow `local_ip` / `local_port` / protocol / byte counts; node degree or age.

### 1.7 Windowing and splits

```python
# build_orthrus_graphs.py:107-146
start, end = cfg.dataset.start_end_day_range
for day in range(start, end):
    date_start = cfg.dataset.year_month + '-' + str(day) + ' 00:00:00'
    ...
    sql = """select * from event_table where timestamp_rec>'%s' and timestamp_rec<'%s'
             ORDER BY timestamp_rec, event_uuid;"""
    ...
    BATCH = 1024
    window_size_in_sec = cfg.graph_construction.build_graphs.time_window_size * 60_000_000_000
    if batch_edges[-1][-2] > start_time + window_size_in_sec:
```

- **Every day in `start_end_day_range` is built** (E3: 2–13, E5: 8–17). Splits are applied
  later, by selecting graph directories.
- Day boundaries at **US/Eastern midnight** (`provnet_utils.py:93`), strict `>` / `<`.
- Ordering is `timestamp_rec, event_uuid` — deterministic tie-break.
- Windows are **batch-quantized**: a window closes at the first 1024-edge boundary
  *after* `time_window_size` (15.0 min) has elapsed, so windows are ≥15 min with
  data-dependent length. The variable is misnamed `window_size_in_sec`; the units are ns.
- ⚠️ **The trailing partial window of each day is silently discarded.** The cut condition
  has no "last batch" branch, so whatever remains in `temp_list` when the day's events run
  out is never saved.

**Edge fusion is unconditional** (`build_orthrus_graphs.py:176-197`) — there is no flag.
Within a window, edges are grouped by `(src, dst)`, sorted by time, and each **run of
consecutive identical operations collapses to one edge** carrying the run's first
timestamp and `event_uuid`.

**Splits** (`src/config.py:143-260`) are lists of graph directories:

| Dataset | train | val | test | unused |
| --- | --- | --- | --- | --- |
| CADETS_E3 | 3, 4, 5, 7, 8, 9, 10 | 2 | 6, 11, 12, 13 | — |
| THEIA_E3 | 2, 3, 4, 5 | 9 | 10, 12, 13 | 11 |
| CLEARSCOPE_E3 | 3, 4, 5, 7, 8, 9, 10 | 2 | 11, 12 | 6, 13 |
| CADETS_E5 | 8, 9, 11 | 12 | 16, 17 | 10, 13, 14, 15 |
| THEIA_E5 | 8, 9, 10 | 11 | 14, 15 | 12, 13, 16, 17 |
| CLEARSCOPE_E5 | 8, 9 | 11 | 14, 15, 17 | 10, 12, 13, 16 |

All: `num_node_types = 3`, `num_edge_types = 10`, window 15.0 min.
Test days are **not** strictly after train days in the E3 datasets — CADETS_E3 trains on
days 3–10 and tests on day 6; CLEARSCOPE_E3 does the same.

---

## 2. Per-dataset detail

### 2.1 CADETS E3 — `src/create_database/cadets_e3.py`

- Pre-pass registers `Subject` and `FileObject` uuids with value `None`
  (`:9-26`); `FILE_OBJECT_UNIX_SOCKET` is skipped here and routed to netflow instead.
- **Subject** (`:89-126`): uuid + `exec` scraped from Event lines. `path` is always
  `None`; `cmd` holds the bare executable name.
- **File** (`:128-165`): path from `predicateObjectPath` on Event lines, gated by
  `'"predicateObjectPath":null,' not in line and '<unknown>' not in line`. The
  `<unknown>` test is a **whole-line** substring test. Last matching event wins. Files
  whose path never resolves are stored with `path = None` rather than dropped.
- **Netflow** (`:29-87`): full 4-tuple regex, plus the unix-socket branch with all-`None`
  addresses.
- On regex failure in `store_subject`, the fallback assigns the **string** `"null"`
  where the success path assigns a 2-element list (`:105-108`); the insert then evaluates
  `[i] + [hash] + "null" + [index_id]`, which raises `TypeError` (`list + str`).

### 2.2 CADETS E5 — `cadets_e5.py`

Same structure as CADETS E3 with `cdm20` namespaces. Differences:

- The file-path extraction has **no `<unknown>` / null gate** — `predicateObjectPath` is
  taken whenever present, else `None`.
- Same `"null"`-string fallback crash in `store_subject`.

### 2.3 THEIA E3 — `theia_e3.py`

- Subject from real `Subject` records; `path` from
  `'"properties":{(.*?)"path":"(.*?)","ppid"'`, `cmd` from `cmdLine`. Both default to
  `None` when absent — the node is kept either way.
- File keyed on `"filename"`.
- ⚠️ Netflow remote address/port are uuid characters (§1.4).

### 2.4 CLEARSCOPE E3 — `clearscope_e3.py`

- Subject `path` is hardcoded `None` (comment: *"no path for subject nodes on
  clearscope_e3"*).
- ⚠️ Subject `cmd` is `subject_cmd[0][0]` — the **first character** of the command line.
- File keyed on `"path"`.
- ⚠️ Netflow remote address/port are uuid characters.

### 2.5 THEIA E5 / CLEARSCOPE E5 — `theia_e5.py`, `clearscope_e5.py`

- THEIA E5: subject `path` from the Subject regex, `cmd` from `cmdLine`; file on `"path"`.
- CLEARSCOPE E5: subject `path` hardcoded `None`, `cmd` from `cmdLine` (correctly indexed).
- ⚠️ Both share the netflow uuid-character bug.

---

## 3. Cross-dataset comparison

### 3.1 What a node *is*

| Dataset | subject identity | subject `path` | subject `cmd` | file label | netflow `remote_ip` |
| --- | --- | --- | --- | --- | --- |
| CADETS E3 | `sha256(uuid)` | always `None` | `exec` | `predicateObjectPath` (gated) | correct |
| CADETS E5 | `sha256(uuid)` | always `None` | `exec` | `predicateObjectPath` | correct |
| THEIA E3 | `sha256(uuid)` | `properties.path` | `cmdLine` | `filename` | ⚠️ uuid char |
| THEIA E5 | `sha256(uuid)` | Subject `path` | `cmdLine` | `path` | ⚠️ uuid char |
| CLEARSCOPE E3 | `sha256(uuid)` | always `None` | ⚠️ first char of `cmdLine` | `path` | ⚠️ uuid char |
| CLEARSCOPE E5 | `sha256(uuid)` | always `None` | `cmdLine` | `path` | ⚠️ uuid char |

Since the model's subject feature is `path + ' ' + cmd` (§1.6), CADETS subjects are
featurized as `"None <exec>"` and CLEARSCOPE E3 subjects as `"None <single char>"`.

### 3.2 Edges

Identical across all six: blacklist of 4 at import, `rel2id` (10) at construction, shared
10-entry reversal list, subject→subject kept, `event_uuid` present.

---

## 4. Catalogue of exclusions

| Rule | Where | Effect |
| --- | --- | --- |
| `exclude_edge_type` (4 types) | import | FCNTL / OTHER / ADD_OBJECT_ATTRIBUTE / FLOWS_TO never stored |
| `include_edge_type = rel2id` (10 types) | `build_orthrus_graphs.py:99` | everything else dropped at construction, incl. `EVENT_CLOSE`, `EVENT_MMAP`, `EVENT_LSEEK` |
| `len(i) != 64` | all node inserts | selects uuid keys out of the bidirectional dicts |
| both endpoints must resolve | `store_event` | events touching un-imported entities dropped |
| `'<unknown>' not in line`, `'"predicateObjectPath":null,' not in line` | CADETS E3 only | file path suppressed by a whole-line substring test |
| `FILE_OBJECT_UNIX_SOCKET` | CADETS E3/E5 | excluded from the file table, added to netflow with `None` addresses |
| unconditional edge fusion | `build_orthrus_graphs.py:176-197` | consecutive same-type edges collapsed |
| no last-batch flush | `build_orthrus_graphs.py:146` | trailing partial window of each day discarded |
| `unused_files` | split config | days built but never loaded |
| `NUM_TEST_EDGES = 2000` | `build_orthrus_graphs.py:225-226` | unit-test path truncates graphs |

**Silent-failure surface**: bare `except: pass` or `except: fail_count += 1` in every
parser; `fail_count` is incremented in some modules and **never logged or asserted**
anywhere. There is no record of how many raw records were dropped.

---

## 5. Findings that can change downstream outcomes

### 5.1 Node identity is the UUID — one node per entity

`sha256(uuid)` everywhere (§1.3). Every process instance is its own node with its own
memory slot and neighbour list; process lineage survives. This is the semantics a
temporal model assumes, and it is what makes ORTHRUS a usable reference point for a
memory-based comparison.

### 5.2 ⚠️ Netflow remote address is corrupt in four of six datasets

`res[0][0]` / `res[0][1]` index into the uuid string (§1.4). ORTHRUS features netflow
nodes on `remote_ip` alone (`use_port: False`), so for THEIA E3/E5 and CLEARSCOPE E3/E5
every netflow node's entire feature content is **one hex character** — roughly 16 distinct
netflow features across the dataset. Any result involving network-side detection on those
four datasets should be treated as unsupported until re-run with the field fixed.

### 5.3 ⚠️ Subject features are degenerate in three of six datasets

CADETS E3/E5 store `path = None` and feed `"None <exec>"`; CLEARSCOPE E3 additionally
truncates the command line to its first character (§3.1). Only THEIA E3/E5 and
CLEARSCOPE E5 carry a meaningful process label.

### 5.4 Unconditional edge fusion destroys event multiplicity

§1.7. A process reading a file 10 000 times in a window becomes one edge. Edge counts are
not comparable to the raw logs, burst/frequency signal is removed, and a temporal model
sees one update where the log had thousands. There is no flag to turn this off.

### 5.5 Windows are batch-quantized, and the day's tail is dropped

A window closes at the first 1024-edge boundary after 15 minutes, and the final partial
window of every day is never written (§1.7). Both the window length distribution and the
per-day coverage are therefore artifacts of the batching, not of the configured window
size.

### 5.6 Two label pipelines, only one of which reaches the model

`use_hashed_label: True` hashes the graph node label, but features come from a separate
`get_indexid2msg` read that ignores it (§1.6). Reading `config/orthrus.yml` alone gives
the wrong picture of what the model sees — `node_label_features` does **not** control the
input features.

### 5.7 Featurization is trained on the whole database

The word2vec corpus covers every node in the DB with no split filter
(`build_feature_word2vec.py:10-30`). Unsupervised and label-free, so not label leakage,
but strictly transductive: test-set node labels are in the vocabulary. An inductive
comparison needs a corpus restricted to train.

### 5.8 The importable file set is whatever matched a glob

`glob.glob(f"{filepath}/*json*")` (§1). Two runs over differently-populated directories
produce different databases with no record of the difference.

### 5.9 Test days are not strictly after train days

CADETS_E3 trains on days 3–10 and tests on day 6; CLEARSCOPE_E3 likewise (§1.7).
Defensible for a per-window anomaly detector, but the split is not chronological and does
not support claims about temporal generalization.

---

## 6. Downstream filters that also shape the results

- **Ground truth**: node-UUID CSVs under `Ground_Truth/darpa/<E3|E5>-<DATASET>/…csv`,
  listed per dataset in `ground_truth_relative_path`, plus per-attack windows in
  `attack_to_time_window` (`src/config.py:143-260`). Two THEIA E3 attack files are
  commented out with reasons — *"attack failed so we don't use it"* and *"attack only at
  network level, not system"* — i.e. the GT set is curated.
- **Detection** is node-level (`src/detection/node_evaluation.py`) over the
  `predict_edge_type` objective (`config/orthrus.yml:56-60`).
- **Attack reconstruction** (`src/attack_reconstruction/tracing.py`, DepImpact) runs on
  flagged nodes and is a separate filtering stage.
- I found **no equivalent of KAIROS's hard-coded `is_include_key_word` path lists** in
  this repo.

---

## 7. Reproduction note

Not verified experimentally in this pass. Worth writing as targeted checks:

1. Dump `netflow_node_table.dst_addr` for THEIA E3 and assert it is an IP — this
   confirms §5.2 in one query and sizes the damage.
2. Count distinct `subject_node_table.cmd` values for CLEARSCOPE E3 — §5.3 predicts a
   very small alphabet.
3. Count edges before and after fusion on one day (§5.4).
4. Compare the number of windows written per day against
   `ceil(day_events / 1024)` to quantify the dropped tail (§5.5).
