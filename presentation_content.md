# 📊 Compiled Minds — Hackathon Presentation Content

This document contains the complete, pre-filled text for each slide of the **H2S INDIA.RUNS** presentation template. You can copy and paste this text directly into your slides.

---

## 🖥️ Slide 1: Title Slide
* **Slide Title**: Build what next India runs on
* **Team Name**: Compiled Minds
* **Team Leader Name**: P.SAI HARSHA
* **Problem Statement**:
  > **Intelligent Candidate Discovery & Ranking Challenge**
  > Building a production-ready, decoupled four-stage candidate retrieval and ranking pipeline that achieves high factual recall, handles adversarial profiles (cheats/spam/honeypots), and processes 100K candidates under tight latency and hardware constraints (CPU-only, 16GB RAM, offline execution).

---

## 🖥️ Slide 2: Solution Overview
* **What is your proposed solution?**
  * We designed and implemented a **Decoupled 4-Stage Hybrid Pipeline** optimized for CPU throughput and reliability:
    * **Layer 1 (Coarse Filter)**: Memory-efficient streaming parser with a bounded min-heap (capped at 1,500 candidates) applying strict geographic and career filters.
    * **Layer 2 (Semantic Relevance)**: Offline, INT8-quantized SentenceTransformer (`all-MiniLM-L6-v2`) scoring.
    * **Layer 3 (Behavioral Calibration)**: Continuous trust score scoring derived from 13 metadata checks (easy-apply spam bots, exam cheating, connection farming).
    * **Layer 4 (Output Reasoning)**: Deterministic, zero-hallucination explanation engine appending security warnings.
* **What differentiates your approach from traditional candidate matching systems?**
  * **Decoupled Architecture**: Keeps expensive vector embeddings bounded to the top 1,500 candidates, preventing OOM.
  * **Honeypot-Resistant**: Naturally screens out impossible signup dates and chronological profile anomalies.
  * **Behavioral Modeling**: Scores availability and activity metrics (e.g. response rates, connection farming) rather than just keyword matching.

---

## 🖥️ Slide 3: JD Understanding & Candidate Evaluation
* **What are the key requirements extracted from the JD?**
  * **Experience**: 5–9 years of experience in NLP, IR, vector databases (FAISS, Milvus, OpenSearch), and ranking metrics (NDCG, MAP, MRR).
  * **Location**: Noida or Pune preferred. Hyderabad, Mumbai, and Delhi NCR commuter hubs welcome.
  * **Exclusions**: Pure research backgrounds, generic LangChain-only developers, and strict IT outsourcing timelines (unless product experience exists).
* **Which candidate signals are most important for determining relevance? / How does your solution evaluate candidate fit beyond keyword matching?**
  * **Commute Haversine Distance**: Checks geographic proximity rather than string-matching cities.
  * **IT Services Trap Bypass**: Allows candidates currently working at outsourcing firms if their history shows product-company engineering experience.
  * **Trust Calibration**: Uses recruiter response rates, connection-to-endorsement ratios, and assessment integrity flags to penalize bot-like activity.

---

## 🖥️ Slide 4: Ranking Methodology
* **How does your system retrieve, score, and rank candidates?**
  * **Step 1**: Streams candidates sequentially from JSONL to conserve memory, filtering out hard exclusions.
  * **Step 2**: Bounded Heap keeps top 1,500 coarse scorers for embedding representation.
  * **Step 3**: MiniLM transformer scores semantic similarity against the Job Description.
  * **Step 4**: Multiplies semantic score by YOE compatibility offset and behavioral trust multiplier.
* **What models, algorithms, or heuristics are used?**
  * **Model**: `all-MiniLM-L6-v2` transformer dynamically quantized to INT8.
  * **Distance Algorithm**: Haversine distance mapping to calculate commuting distance (capped at 100km).
  * **Heuristics**: Continuous scaling function for Years of Experience fits (Gaussian decay outside ideal range).
* **How are multiple candidate signals combined into a final ranking?**
  * $$\text{Final Score} = \text{Base Relevance} \times \text{Semantic Score} \times \text{YOE Offset} \times \text{Behavioral Trust Score}$$
  * Equal scores are broken deterministically by ascending alphabetical order of `candidate_id`.

---

## 🖥️ Slide 5: Explainability & Data Validation
* **How are ranking decisions explained?**
  * A rule-based **Deterministic Reason Generator** writes a 1–2 sentence reasoning using concrete metrics extracted from the candidate profile (e.g., specific years of experience, current title match, trust level, location proximity).
* **How do you prevent hallucinations or unsupported justifications?**
  * **Zero LLM Generation**: Reasoning is constructed dynamically using a deterministic template populated directly from verified profile data, guaranteeing that no skills, employers, or metrics are hallucinated.
* **How does your solution handle inconsistent, low-quality, or suspicious profiles?**
  * **Continuous Trust Score**: Drop trust from 1.0 down to 0.01 based on 13 anomalies (e.g. instant replies, connection-farming bots, skipping interviews).
  * **Honeypot Filter**: Detects impossible timelines (e.g. 10 years experience at a 2-year old company) and filters them out.

---

## 🖥️ Slide 6: End-to-End Workflow

```
[Raw candidates.jsonl] (100K)
          │
          ▼  (Memory-efficient streaming)
┌──────────────────────────────────────────────┐
│  Layer 1: Hard Filters & NCR Commute Check  │
└──────────────────────────────────────────────┘
          │ (Heap Filter - Top 1,500)
          ▼
┌──────────────────────────────────────────────┐
│  Layer 2: INT8 Quantized Sentence Embedding │
└──────────────────────────────────────────────┘
          │ (Semantic Relevance Math)
          ▼
┌──────────────────────────────────────────────┐
│  Layer 3: 13 Anomaly Behavioral Trust Score   │
└──────────────────────────────────────────────┘
          │ (Trust Multipliers Applied)
          ▼
┌──────────────────────────────────────────────┐
│  Layer 4: Deterministic Reason Generator     │
└──────────────────────────────────────────────┘
          │ (Tie-Break resolved)
          ▼
[Final CSV Output] (Exactly Top 100)
```

---

## 🖥️ Slide 7: System Architecture
* **Decoupled Architecture**:
  * **Frontend (Sandbox)**: Streamlit Web UI hosted on Streamlit Cloud for interactive sample testing.
  * **Core Pipeline Engine**: Decoupled modules (`ranking_pipeline.py`) executing sequentially or via multiprocessing depending on the host CPU cores.
  * **Model Layer**: Local, offline SentenceTransformer model files loaded directly from `./local_model` with network access disabled.
  * **Cache Layer**: Pickled vector embeddings database (`.embeddings_cache.pkl`) allowing incremental embedding and rapid warm-starts.

---

## 🖥️ Slide 8: Results & Performance
* **What results or insights demonstrate ranking quality?**
  * **100% Format Compliance**: Passes the official validator test with no duplicate ranks or candidate IDs.
  * **Honeypot Rate**: 0% in the top 100 rankings (system naturally avoids impossible trap candidates).
  * **Accurate Recruiter Relevance**: Combines behavioral flags so that inactive or bot candidates are down-weighted.
* **How does your solution meet the challenge’s runtime and compute constraints?**
  * **Execution Speed**:
    * **Cold Start**: **`41.46 seconds`** (includes initializing model weights)
    * **Warm Start (Cached)**: **`19.04 seconds`**
  * **Throughput**: **`8,000+ candidates/second`** (Warm)
  * **RAM Footprint**: Capped at **`568 MB`** (completely OOM-safe, well under the 16GB limit).

---

## 🖥️ Slide 9: Technologies Used
* **Languages & Core**: Python (gzip, json, heapq, hashlib, re)
* **Embeddings & ML**: PyTorch, Hugging Face `transformers`, `sentence-transformers` (all-MiniLM-L6-v2)
* **Data Processing & Utilities**: NumPy, Pandas, special characters normalization (`unicodedata`)
* **Testing & Diagnostics**: unittest, psutil (RAM/CPU benchmarking)
* **Sandbox Web App**: Streamlit (Community Cloud)

---

## 🖥️ Slide 10: Submission Assets
* **GitHub Repository**: [Compiled-Minds](https://github.com/Harsha2134/Compiled-Minds)
* **Hosted Sandbox UI**: [compiled-minds-redrob.streamlit.app](https://compiled-minds-redrob.streamlit.app/)
* **reproduce_command**:
  ```bash
  python ranking_pipeline.py --candidates ./candidates.jsonl --out ./submission.csv
  ```
