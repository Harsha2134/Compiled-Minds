# Redrob Candidate Ranking Pipeline

This repository contains the production-ready implementation of the candidate discovery and ranking system designed for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

---

## 🏗️ Architecture Summary

Our system implements a decoupled **four-stage hybrid pipeline** optimized for CPU throughput, factual recall, and platform credibility:

1. **Layer 1: Hard Filters & Exclusions**: Screens the candidate pool using memory-efficient JSONL streaming and a bounded min-heap (`heapq` capped at 1,500 candidates). Disqualifies candidates based on location commuter hub rules (Noida, Pune, Delhi NCR, Mumbai, Hyderabad), pure research backgrounds, IT outsourcing firm timelines, and temporal signup date honeypots.
2. **Layer 2: Core Semantic Relevance**: Computes semantic similarity on the top 1,500 survivors using an offline, dynamically quantized (INT8) `SentenceTransformer` (MiniLM-L6-v2) model. Applies Years of Experience (YOE) range offsets and job hopper penalties.
3. **Layer 3: Behavioral Calibration (Powerlayer)**: Calibrates relevance scores using a continuous Trust Score ($1.0 \to 0.0$) derived from 13 metadata checks (assessment cheating proxies, easy-apply spam bots, and open-to-work availability contradictions).
4. **Layer 4: Output Reasoning**: Generates deterministic, hallucination-free explanations and appends security audit warnings (e.g. flagged reviews) to the final output.

---

## 🚀 Getting Started & Reproduction

### 1. Environment Setup
Set up the virtual environment and install the required dependencies:
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution Command
To run the ranking pipeline end-to-end on the candidate database and generate the submission file, run the following single command:
```bash
python ranking_pipeline.py --candidates ./candidates.jsonl --out ./submission.csv
```

* **Warm Start (Cached)**: **`19.04 seconds`**
* **Cold Start (First Run)**: **`41.46 seconds`**
* **RAM Footprint**: Bounded under **`568 MB`** (completely OOM-safe)
* **Throughput**: **`5,251.79 candidates/second`** (Warm)

---

## 🛡️ Validation

Run the official hackathon validator on the output file to verify compliance:
```bash
python validate_submission.py submission.csv
```

All 40 unit tests are green:
```bash
python -m unittest test_ranking_layers.py
```
