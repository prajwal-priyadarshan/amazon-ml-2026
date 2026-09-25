# Amazon ML Challenge 2026 — Business Entity Resolution
## Complete Dataset Analysis & Model Training Strategy

---

## 1. Problem Overview

**Task:** Given business records from 3 independent, noisy data sources, find all records in Source 2 and Source 3 that refer to the same real-world business as each Source 1 (reference) entity.

**Source 1 is the deduplicated reference.** Output must map every S1 entity → list of matching S2/S3 entities (empty = singleton).

**Evaluation:** Macro-averaged **F₀.₅ score** — precision-weighted 2× over recall.

---

## 2. Dataset Statistics

### 2.1 Row Counts

| File | Rows |
|---|---|
| train_source1.tsv | **2,206,821** |
| train_source2.tsv | **5,034,616** |
| train_source3.tsv | **5,285,603** |
| train_ground_truth.tsv | **2,206,821** |
| test_source1.tsv | **1,732,544** |
| test_source2.tsv | **4,887,273** |
| test_source3.tsv | **5,082,316** |
| **Total entities (train)** | **~12.5M** |
| **Total entities (test)** | **~11.7M** |

> [!IMPORTANT]
> This is a massive-scale dataset (~12.5M train, ~11.7M test). Naive O(n²) pairwise comparison is infeasible (~70 trillion pairs). **Blocking/candidate generation is the critical bottleneck.**

### 2.2 Country Distribution

| Country | Train S1 | Train S2 | Train S3 | Test S1 | Test S2 | Test S3 |
|---|---|---|---|---|---|---|
| **US** | 1,323,633 | 3,016,817 | 3,170,056 | 663,106 | 1,871,330 | 1,945,701 |
| **India** | 883,188 | 2,017,799 | 2,115,547 | 809,986 | 2,312,565 | 2,405,000 |
| **France** | ❌ 0 | ❌ 0 | ❌ 0 | **259,452** | **703,378** | **731,615** |

> [!WARNING]
> **France is a zero-shot country** — it appears ONLY in the test set (259k S1 entities). The pipeline must generalize without France training examples. This makes language-agnostic features (character n-grams, transliteration-robust embeddings) critical.

### 2.3 Missing Values

| Source | business_name nulls | business_address nulls |
|---|---|---|
| S1 (reference) | **0** | **0** |
| S2 | 2 | **168,967 (3.4%)** |
| S3 | 13 | **175,916 (3.3%)** |

> [!NOTE]
> Source 1 has no missing values. S2/S3 both have ~3.4% missing addresses — name-only matching must be robust enough to handle these cases.

### 2.4 Field Length Stats (sampled)

| Source | Name avg len | Name max | Address avg len | Address max |
|---|---|---|---|---|
| S1 | 24.0 | 71 | 52.1 | 222 |
| S2 | 25.1 | 104 | 46.2 | 202 |
| S3 | 25.2 | 80 | 46.8 | 198 |

---

## 3. Ground Truth Analysis

### 3.1 Match Distribution

| Matches per S1 entity | Count |
|---|---|
| **0 (singletons)** | **123,247 (5.6%)** |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |
| **≥1 matches** | **2,083,574 (94.4%)** |
| **Max matches** | **11** |
| **Avg matches** | **3.46** |

- **94.4% of S1 entities have at least one match** → most entities are linked
- Typical cluster size is **4–5 records total** (1 S1 + ~3.5 S2/S3)
- S2 contributes **3,693,619 total matches**, S3 contributes **3,944,746**

### 3.2 Source Relationship Structure

```
S1 (deduplicated reference)  ←→  S2 (noisy copy)
       ↑                               ↑
       └──────────── S3 (noisy copy) ──┘
```

- S1 is the **ground truth anchor** — deduplicated, clean
- S2 and S3 are **independently generated noisy representations** of the same real-world entities
- A single S1 entity can match **multiple S2 AND multiple S3 records simultaneously**
- S2 and S3 are **not directly matched to each other** — only to S1

---

## 4. Noise Pattern Analysis

From observed matched pairs, we see the following systematic noise patterns:

### 4.1 Name Noise Patterns

| Pattern | Example |
|---|---|
| **Script/language transliteration** | S1: "Raj Investments LLP" ↔ S2: "राज इन्वेस्टमेंट एलएलपी" (Hindi script) |
| **Legal suffix variation** | "Maure Williams Colombier Inc" ↔ "Maure Williams Colombier" (no suffix) |
| **Typos** | "Dahlia Power Reliable Scientific LLC" ↔ "Dahlia Power Reliable Scientific" + character swaps |
| **Word reordering** | "Hendricks and Flowers Inc" ↔ "Hendricks and Inc Flowers" |
| **Abbreviation** | "Crystal Staffing Solutions LLC" ↔ "CRYSTAL STAFFING SOLUTIONS-L.L.C." |
| **Partial names** | "Dick Regional Armada Corp" ↔ "Dick Regional" (truncated) |
| **Case variation** | "Obsidian, LLC" ↔ "obsidian, llc" |
| **Diacritics** | "Payne Enterprises" ↔ "Payne Ènterprises" |
| **DBA/Trade name** | "Uptown Pub" ↔ "Solkeloquo" (completely different DBA name!) |
| **Phone numbers embedded** | "Chordia + Pagnters - 7306204978" |
| **Repeated words** | "Orellana Investments Investments Llc" |
| **Prefix insertion** | "AP Hospitality Inc" ↔ "AP INC SERVICE #80430" |

> [!CAUTION]
> The most challenging case is **DBA/trade name variation** (e.g., "Uptown Pub" ↔ "Solkeloquo"). Only address matching can resolve this. Pure name-matching will fail here.

### 4.2 Address Noise Patterns

| Pattern | Example |
|---|---|
| **Abbreviation** | "Fremont Street" ↔ "FREMONT ST" |
| **Component reordering** | "3315 Fremont Street, Peoria, IL" ↔ "IL, FREMONT STREET, PEORIA" |
| **Missing components** | Full address ↔ just city + state |
| **Typos** | "45th Terrace" ↔ "45ND TERRACE" |
| **Spelling errors** | "Buisness Centre" (typo) |
| **State code vs. full** | "Karnataka" ↔ "KA" |
| **Number format** | "AF-684" ↔ "AF-0684" |
| **Missing (NaN)** | ~3.4% of S2/S3 have no address at all |
| **Script mixing** | Address partially in local script (Hindi/Telugu) |
| **Extra fields** | Phone numbers embedded in address |

### 4.3 Multi-Script Challenge

India records frequently appear with:
- S1 (clean): English transliteration
- S2 (noisy): Native script (Hindi/Telugu/Tamil/Kannada/Malayalam)
- S3: Mixed or abbreviated English

This is extremely difficult for string similarity — requires **multilingual embeddings** (e.g., mBERT, LaBSE, or IndicBERT).

---

## 5. Model Training Strategy

### 5.1 Pipeline Architecture

```
[Source 1]  [Source 2]  [Source 3]
     │            │            │
     └────────────┴────────────┘
                  │
           ┌──────▼──────┐
           │  Blocking /  │   ← Candidate Generation
           │  Indexing    │     (reduces ~70T pairs → manageable set)
           └──────┬───────┘
                  │
           ┌──────▼──────┐
           │  Feature     │   ← Similarity Feature Engineering
           │  Engineering │
           └──────┬───────┘
                  │
           ┌──────▼──────┐
           │  Matching    │   ← Binary Classifier (Match / No Match)
           │  Model       │
           └──────┬───────┘
                  │
           ┌──────▼──────┐
           │  Threshold + │   ← Post-processing & Deduplication
           │  Output      │
           └─────────────┘
```

---

### 5.2 Stage 1: Blocking / Candidate Generation

**Goal:** Reduce the search space from O(S1 × S2 + S1 × S3) ~= 20B pairs down to a manageable candidate set while maximizing recall.

#### Blocking Methods (use ensemble of multiple strategies):

1. **TF-IDF / BM25 on combined name+address text**
   - Build TF-IDF index for all S2/S3 records
   - For each S1 entity, retrieve top-K (e.g., K=20) most similar candidates
   - Use `sklearn.TfidfVectorizer` + sparse matrix multiplication or `faiss` for ANN

2. **Token-level inverted index**
   - Index on name tokens (words, character n-grams)
   - Find S2/S3 records sharing ≥1 token with S1 name

3. **Country-partitioned blocking**
   - ALWAYS block within the same country first (country=US, India, France)
   - Massive reduction with no false negatives (matches don't cross countries)

4. **Minhash LSH**
   - `datasketch` library: compute MinHash signatures on character 3-grams
   - Very fast approximate matching even for transliterated strings

5. **Phonetic blocking**
   - `soundex` / `metaphone` / `Double Metaphone` for English names
   - Group by phonetic key, then do pairwise matching within blocks

6. **Embedding-based ANN retrieval** (FAISS)
   - Use multilingual sentence embeddings (LaBSE or `paraphrase-multilingual-MiniLM-L12-v2`)
   - Encode all records, retrieve top-K nearest neighbors via FAISS
   - Handles the script/transliteration problem

**Recommended blocking target:** Achieve **≥90% candidate recall** (every true pair is in candidates) while reducing pairs to ≤200–500 candidates per S1 entity.

---

### 5.3 Stage 2: Feature Engineering

For each candidate pair (S1_i, S2/S3_j), compute these features:

#### Name Similarity Features
| Feature | Implementation |
|---|---|
| **Jaccard similarity** (word-level) | `|A∩B| / |A∪B|` on tokenized names |
| **Jaccard similarity** (char 3-gram) | Character trigram overlap |
| **Levenshtein / edit distance** | `python-Levenshtein` |
| **Jaro-Winkler** | `jellyfish` library |
| **TF-IDF cosine** | Precomputed from blocking step |
| **Token set ratio** | `fuzzywuzzy / rapidfuzz` |
| **Common word count** | After stopword removal |
| **Name embedding cosine** | Sentence transformer similarity |

#### Address Similarity Features
| Feature | Implementation |
|---|---|
| **Jaccard (word-level)** | Same as name |
| **Jaccard (char 3-gram)** | Character trigram overlap |
| **Levenshtein** | Edit distance on full address |
| **Postal code match** | Exact match on extracted postal codes |
| **City/state match** | Extracted city, state, country code |
| **TF-IDF cosine** | Address TF-IDF cosine similarity |
| **Address embedding cosine** | Multilingual embedding similarity |
| **Address length ratio** | Ratio of lengths (handles truncation) |

#### Structural/Meta Features
| Feature | Implementation |
|---|---|
| **Country match** | Binary (should always be 1 after country blocking) |
| **Name length ratio** | `len(s1_name) / len(s2_name)` |
| **Address is null (S2/S3)** | Binary flag |
| **Name token overlap count** | # of shared tokens |
| **Is singleton candidate** | Heuristic score for no-match prediction |

---

### 5.4 Stage 3: Matching Model

#### Recommended: Gradient Boosting (LightGBM/XGBoost) + Fine-tuned Transformer Reranker

**Phase A — Fast LightGBM classifier (all candidates)**

```python
# Binary classification: is this pair a true match?
# Label 1 = match (from ground truth), 0 = non-match
# Train with heavy class imbalance handling (neg:pos ratio ~10:1 to 100:1)
model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=127,
    scale_pos_weight=10,  # adjust based on actual imbalance
    subsample=0.8,
    colsample_bytree=0.8
)
```

**Phase B — Cross-encoder reranker (top candidates only)**

For high-confidence candidates from Phase A, use a cross-encoder:
```python
# Model: cross-encoder/ms-marco-MiniLM-L-6-v2 or fine-tuned on this data
# Input: "[CLS] {s1_name} [SEP] {s1_address} [SEP] {s2_name} [SEP] {s2_address} [SEP]"
# Output: match score (0–1)
# Fine-tune on positive pairs from ground truth + hard negatives from blocking
```

**Why precision-heavy threshold matters:**
- F₀.₅ penalizes false positives 2× more than false negatives
- Set decision threshold higher than 0.5 — tune on validation set to maximize F₀.₅
- Typical optimal threshold: **0.6–0.75**

#### Model License Constraint
> [!IMPORTANT]
> Final model must be **MIT or Apache 2.0 licensed** and **≤8B parameters**.
> - ✅ LightGBM, XGBoost, CatBoost (MIT/Apache)
> - ✅ sentence-transformers: `paraphrase-multilingual-MiniLM-L12-v2` (Apache 2.0, ~117M params)
> - ✅ LaBSE (Apache 2.0, ~471M params)
> - ✅ `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache 2.0)
> - ✅ Multilingual BERT (Apache 2.0, ~178M params)
> - ❌ LLaMA, Mistral (custom licenses — do NOT use)

---

### 5.5 Training Data Construction

#### Positive Examples
- Directly from `train_ground_truth.tsv`: each (S1_id, S2/S3_id) pair = positive label

#### Negative Examples (Critical for quality)
- **Random negatives**: Random S2/S3 records from same country that are NOT matches
- **Hard negatives**: Candidates retrieved by blocking that are NOT true matches (most valuable!)
- **Ratio**: 5–20 negatives per positive (tune based on validation F₀.₅)

#### Data Volume
- ~7.6M positive pairs (3.69M S2 + 3.94M S3)
- Target: ~40–80M training examples (positives + negatives)

---

### 5.6 Handling Special Cases

#### Singletons (no match)
- 5.6% of S1 entities have no matches
- For singletons: blocking must still run but no candidates should pass the threshold
- Explicitly train with singleton S1 entities' blocking candidates as hard negatives
- Correctly predicting singletons = free **1.0 F₀.₅** per entity

#### Multi-script (Hindi, Telugu, Tamil, Kannada in S2)
- Multilingual embeddings handle this automatically
- Additionally: Unicode transliteration (`indic-transliteration` or `aksharamukha`)
- Character n-gram Jaccard still works across scripts if same characters

#### France (zero-shot)
- Use language-agnostic features: character n-grams, address token overlap, embedding cosine
- LaBSE or `paraphrase-multilingual-MiniLM-L12-v2` covers French natively
- Do NOT hard-code country-specific logic

#### DBA / Trade name cases
- When name similarity is low, rely on address similarity to find matches
- Embed address independently and use address-embedding cosine as a fallback signal

---

### 5.7 Validation Strategy

- Hold out **10% of S1 entities** (stratified by country + match count) as validation
- Compute F₀.₅ per entity, then macro-average
- Use validation to:
  1. Tune blocking recall (K candidates per entity)
  2. Tune LightGBM hyperparameters
  3. Tune decision threshold for maximum F₀.₅

```python
# F_0.5 formula
def f05(precision, recall):
    return (1.25 * precision * recall) / (0.25 * precision + recall + 1e-9)
```

---

## 6. Recommended Tech Stack

| Component | Tool |
|---|---|
| Data loading | `pandas`, `polars` (faster for large files) |
| TF-IDF indexing | `sklearn.TfidfVectorizer` + `scipy.sparse` |
| ANN search | `faiss-cpu` |
| MinHash LSH | `datasketch` |
| String similarity | `rapidfuzz` (fast Levenshtein, Jaro-Winkler) |
| Embeddings | `sentence-transformers` (LaBSE or multilingual-MiniLM) |
| ML model | `lightgbm` + optional `xgboost` ensemble |
| GPU training (if avail.) | `torch` for transformer fine-tuning |
| Validation | Custom F₀.₅ macro-average scorer |

---

## 7. Key Insights & Priority Action Items

1. **Blocking is everything** — no blocking recall = no match recall. This is the ceiling.
2. **Country-partition first** — never compare US ↔ India records.
3. **Multilingual embeddings** are mandatory for the multi-script India data and zero-shot France.
4. **Set threshold at ~0.65+** for F₀.₅ — err on the side of precision.
5. **Hard negatives from blocking** are the most valuable training signal.
6. **Singleton handling**: if blocking returns zero strong candidates, predict no match.
7. **Address is often more reliable than name** for DBA/trade name cases.
8. **France requires language-agnostic features** — test with French business name examples.

---

## 8. Expected Pipeline Timeline

```
Phase 1: Blocking + Candidate Generation   (~1–2 days)
Phase 2: Feature Engineering               (~1 day)
Phase 3: LightGBM Training + Tuning       (~1 day)
Phase 4: Transformer Reranker (optional)   (~1–2 days)
Phase 5: Threshold Tuning + Submission     (~0.5 day)
```
