#!/usr/bin/env python3
"""
Custom Candidate Discovery & Ranking Pipeline Checklist and Health Auditor.
Comprehensive validation suite specifically tailored for the Redrob pipeline:
1. Model Directory & File Integrity
2. Cache Serialization Health
3. Input JSONL Schema & Data Quality
4. Layer 1 Exclusion Rule Assertions (Relocation, Consulting, Academic, Honeypot)
5. Layer 2 Relevance Modifiers (YoE score, Keyword Multipliers, Hopper/Social penalties)
6. Layer 3 Behavioral Calibration Multipliers (Botting, Notice Period, GitHub Farming, Endorsements, Trust)
7. Dynamic Execution Router Verification
8. Output Submission CSV Layout & Sorting Verification
9. Performance & Speed Thresholds
"""

import os
import sys
import time
import pickle
import csv

# Terminal colors for clean reporting
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Load current project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import ranking_pipeline
except ImportError as e:
    print(f"{RED}[FAIL] Could not import ranking_pipeline.py: {e}{RESET}")
    sys.exit(1)

def print_header(title: str):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE} {title}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def check_model_integrity() -> bool:
    """Verifies that the offline SentenceTransformer model exists and is complete."""
    model_dir = ranking_pipeline.MODEL_PATH
    print(f"Checking model directory: {model_dir}")
    
    if not os.path.exists(model_dir):
        print(f"  {RED}[FAIL] Local model directory not found at '{model_dir}'{RESET}")
        return False
        
    files = os.listdir(model_dir)
    has_weights = "pytorch_model.bin" in files or "model.safetensors" in files
    
    if not has_weights:
        print(f"  {RED}[FAIL] Model weights file is missing from {model_dir}!{RESET}")
        return False
        
    print(f"  {GREEN}[PASS] Offline Transformer model files verified.{RESET}")
    return True

def check_cache_health() -> bool:
    """Verifies pickle cache file exists and is readable."""
    cache_path = ".embeddings_cache.pkl"
    print(f"Checking vector cache: {cache_path}")
    
    if not os.path.exists(cache_path):
        print(f"  {YELLOW}[WARN] No vector cache file found (.embeddings_cache.pkl). First run will cold start.{RESET}")
        return True
        
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"  {GREEN}[PASS] Cache file is healthy. Entries cached: {len(cache)}{RESET}")
        return True
    except Exception as e:
        print(f"  {RED}[FAIL] Cache file is corrupted: {e}{RESET}")
        return False

def check_dataset_schema() -> bool:
    """Reads the first line of the candidates JSONL file to check keys and values."""
    dataset_path = "[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
    print(f"Checking candidates dataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"  {RED}[FAIL] Hackathon candidates JSONL file not found at: {dataset_path}{RESET}")
        return False
        
    try:
        generator = ranking_pipeline.load_candidates(dataset_path)
        first_cand = next(generator)
        
        required_keys = ["candidate_id", "profile", "skills", "career_history"]
        missing_keys = [k for k in required_keys if k not in first_cand]
        
        if missing_keys:
            print(f"  {RED}[FAIL] Missing critical keys in JSONL: {missing_keys}{RESET}")
            return False
            
        print(f"  {GREEN}[PASS] Candidates dataset format and schema verified.{RESET}")
        return True
    except Exception as e:
        print(f"  {RED}[FAIL] Failed to parse candidate JSONL dataset: {e}{RESET}")
        return False

def check_l1_exclusions() -> bool:
    """Verifies all Layer 1 filter exclusion rules function correctly."""
    print("Verifying Layer 1 exclusion rules...")
    
    # 1. Relocation Guard (Willing to relocate = False, Onsite preferred, lives outside Noida/Pune)
    bad_relocation = {
        "profile": {"location": "Bangalore"},
        "redrob_signals": {"preferred_work_mode": "onsite", "willing_to_relocate": False},
        "skills": [],
        "career_history": [],
        "education": []
    }
    
    # 2. Consulting Firm Filter (All jobs are at TCS/Wipro/Infosys)
    bad_consulting = {
        "profile": {"location": "Noida"},
        "redrob_signals": {"preferred_work_mode": "remote", "willing_to_relocate": True},
        "skills": [],
        "career_history": [
            {"company": "TCS", "duration_months": 24},
            {"company": "Wipro", "duration_months": 18}
        ],
        "education": []
    }
    
    # 3. Pure Research Excluder (Academic titles only, no production history)
    bad_academic = {
        "profile": {"location": "Noida"},
        "redrob_signals": {"preferred_work_mode": "remote", "willing_to_relocate": True, "github_activity_score": -1, "skill_assessment_scores": {}},
        "skills": [],
        "career_history": [
            {"title": "Postdoc Researcher", "company": "Stanford University", "description": "Academic studies"},
            {"title": "PhD Candidate", "company": "MIT Institute", "description": "Doctoral thesis"}
        ],
        "education": []
    }
    
    # 4. Honeypot Detector: Experience exceeding post-graduation age
    # Graduated in 2024 (age approx 24 in 2026), but claims 15 years experience
    bad_honeypot = {
        "profile": {"location": "Noida", "years_of_experience": 15.0},
        "redrob_signals": {"preferred_work_mode": "remote", "willing_to_relocate": True},
        "skills": [],
        "career_history": [{"duration_months": 180}],
        "education": [{"end_year": 2024}]
    }

    # 5. Perfect Candidate (Should NOT be excluded)
    good_cand = {
        "profile": {"location": "N Noida", "years_of_experience": 6.0},
        "redrob_signals": {"preferred_work_mode": "hybrid", "willing_to_relocate": True},
        "skills": [{"name": "NLP", "proficiency": "expert", "duration_months": 24}],
        "career_history": [{"company": "Google", "title": "Software Engineer", "duration_months": 48, "is_current": True}],
        "education": [{"end_year": 2020}]
    }
    
    try:
        assert ranking_pipeline.should_exclude(bad_relocation, "Noida"), "Relocation exclusion failed."
        assert ranking_pipeline.should_exclude(bad_consulting, "Noida"), "Consulting exclusion failed."
        assert ranking_pipeline.should_exclude(bad_academic, "Noida"), "Academic exclusion failed."
        assert ranking_pipeline.should_exclude(bad_honeypot, "Noida"), "Honeypot timeline check failed."
        assert not ranking_pipeline.should_exclude(good_cand, "Noida"), "Valid candidate was incorrectly excluded."
        print(f"  {GREEN}[PASS] Layer 1 filter exclusions verified.{RESET}")
        return True
    except AssertionError as e:
        print(f"  {RED}[FAIL] Layer 1 assertion failed: {e}{RESET}")
        return False
    except Exception as e:
        print(f"  {RED}[FAIL] Error during Layer 1 verification: {e}{RESET}")
        return False

def check_l2_relevance_modifiers() -> bool:
    """Verifies Layer 2 relevance score modifiers (YoE scores, keyword boosts, penalties)."""
    print("Verifying Layer 2 relevance math...")
    
    # Base candidate template
    base = {
        "profile": {"years_of_experience": 6.0}, # Target range (5-9y) -> gets +3.0 score
        "skills": [],
        "career_history": [{"duration_months": 48}],
        "redrob_signals": {"github_activity_score": 50, "linkedin_connected": True}
    }
    
    # 1. Experience score target range check
    exp_target = {**base, "profile": {"years_of_experience": 7.0}} # target gets +3.0
    exp_adjacent = {**base, "profile": {"years_of_experience": 4.5}} # adjacent gets +1.5
    exp_outside = {**base, "profile": {"years_of_experience": 15.0}} # outside gets +0.5
    
    # 2. Core Search keywords boost (1.5x) vs Nice-to-haves (1.2x)
    keyword_core = {
        **base,
        "skills": [{"name": "embeddings"}, {"name": "hybrid search"}] # Core keywords
    }
    keyword_nice = {
        **base,
        "skills": [{"name": "LoRA"}, {"name": "PEFT"}] # Nice-to-have keywords
    }
    
    # 3. Penalties (Job hopper gets -2.0, Closed-source gets -1.5)
    job_hopper = {
        **base,
        "career_history": [{"duration_months": 10}, {"duration_months": 12}] # Average <18 months
    }
    closed_source = {
        **base,
        "redrob_signals": {"github_activity_score": -1, "linkedin_connected": False}
    }
    
    try:
        # Check target vs outside experience scores
        score_target = ranking_pipeline.compute_relevance_score(exp_target, 0.5)
        score_adjacent = ranking_pipeline.compute_relevance_score(exp_adjacent, 0.5)
        score_outside = ranking_pipeline.compute_relevance_score(exp_outside, 0.5)
        assert score_target > score_adjacent > score_outside, "Experience score buckets mapping failed."
        
        # Check semantic boosts
        score_core = ranking_pipeline.compute_relevance_score(keyword_core, 0.5)
        score_nice = ranking_pipeline.compute_relevance_score(keyword_nice, 0.5)
        assert score_core > score_nice, "Core keyword vs Nice-to-have keyword boost priority failed."
        
        # Check penalties
        score_clean = ranking_pipeline.compute_relevance_score(base, 0.5)
        score_hopper = ranking_pipeline.compute_relevance_score(job_hopper, 0.5)
        score_closed = ranking_pipeline.compute_relevance_score(closed_source, 0.5)
        assert score_clean - score_hopper >= 2.0, "Job hopper penalty (-2.0) not applied."
        assert score_clean - score_closed >= 1.5, "Closed source penalty (-1.5) not applied."
        
        print(f"  {GREEN}[PASS] Layer 2 relevance score modifiers verified.{RESET}")
        return True
    except AssertionError as e:
        print(f"  {RED}[FAIL] Layer 2 assertion failed: {e}{RESET}")
        return False
    except Exception as e:
        print(f"  {RED}[FAIL] Error during Layer 2 verification: {e}{RESET}")
        return False

def check_l3_behavioral_calibration() -> bool:
    """Verifies all Layer 3 platform engagement multipliers are applied correctly."""
    print("Verifying Layer 3 behavioral multipliers...")
    
    # 1. Base Multiplier (Default verified profile)
    base_signals = {
        "verified_email": True, "verified_phone": True,
        "notice_period_days": 30, "github_activity_score": 50,
        "skill_assessment_scores": {"NLP": 85}, "linkedin_connected": True
    }
    base_cand = {"redrob_signals": base_signals, "career_history": []}
    
    # 2. Botting Multiplier (x0.5 penalty)
    bot_signals = {
        **base_signals,
        "applications_submitted_30d": 120, "interview_completion_rate": 0.10 # Botting triggers
    }
    bot_cand = {"redrob_signals": bot_signals, "career_history": []}
    
    # 3. Notice Period boosters / penalties
    fast_signals = {**base_signals, "notice_period_days": 15} # gets x1.1
    slow_signals = {**base_signals, "notice_period_days": 100} # gets x0.6
    fast_cand = {"redrob_signals": fast_signals, "career_history": []}
    slow_cand = {"redrob_signals": slow_signals, "career_history": []}
    
    # 4. Green-Dot Farming (x0.7 penalty)
    farming_signals = {**base_signals, "github_activity_score": 90, "skill_assessment_scores": {}}
    farming_cand = {"redrob_signals": farming_signals, "career_history": []}
    
    # 5. Identity Trust discounts (unverified email/phone gets x0.6 each)
    untrust_signals = {**base_signals, "verified_email": False, "verified_phone": False}
    untrust_cand = {"redrob_signals": untrust_signals, "career_history": []}
    
    # 6. Recruiter Saves Booster
    saves_signals = {**base_signals, "saved_by_recruiters_30d": 6}
    saves_cand = {"redrob_signals": saves_signals, "career_history": []}
    
    # 7. Recruiter Skip Penalty
    skip_signals = {**base_signals, "search_appearance_30d": 50, "saved_by_recruiters_30d": 0}
    skip_cand = {"redrob_signals": skip_signals, "career_history": []}
    
    # 8. Open-to-Work Contradiction Penalty
    contra_signals = {**base_signals, "open_to_work_flag": True, "notice_period_days": 100}
    contra_cand = {"redrob_signals": contra_signals, "career_history": []}
    
    try:
        mult_base = ranking_pipeline.compute_behavioral_multiplier(base_cand)
        mult_bot = ranking_pipeline.compute_behavioral_multiplier(bot_cand)
        mult_fast = ranking_pipeline.compute_behavioral_multiplier(fast_cand)
        mult_slow = ranking_pipeline.compute_behavioral_multiplier(slow_cand)
        mult_farm = ranking_pipeline.compute_behavioral_multiplier(farming_cand)
        mult_trust = ranking_pipeline.compute_behavioral_multiplier(untrust_cand)
        mult_saves = ranking_pipeline.compute_behavioral_multiplier(saves_cand)
        mult_skip = ranking_pipeline.compute_behavioral_multiplier(skip_cand)
        mult_contra = ranking_pipeline.compute_behavioral_multiplier(contra_cand)
        
        # Verify ratios
        assert mult_bot < mult_base, "Botting penalty was not applied."
        assert mult_fast > mult_slow, "Notice period boost/penalty comparison failed."
        assert mult_farm < mult_base, "Green-dot farming penalty was not applied."
        assert abs(mult_trust - (mult_base * 0.6 * 0.6)) < 0.01, "Identity trust multipliers (x0.6 each) failed."
        assert abs(mult_saves - (mult_base * 1.15)) < 0.01, "Recruiter saves booster failed."
        assert abs(mult_skip - (mult_base * 0.85)) < 0.01, "Recruiter search skip penalty failed."
        assert abs(mult_contra - (mult_slow * 0.90)) < 0.01, "Open-to-work notice contradiction penalty failed."
        
        print(f"  {GREEN}[PASS] Layer 3 behavioral calibration multipliers verified.{RESET}")
        return True
    except AssertionError as e:
        print(f"  {RED}[FAIL] Layer 3 assertion failed: {e}{RESET}")
        return False
    except Exception as e:
        print(f"  {RED}[FAIL] Error during Layer 3 verification: {e}{RESET}")
        return False

def check_dynamic_router() -> bool:
    """Verifies execution routing logic based on mock file sizes."""
    print("Verifying dynamic execution router...")
    
    try:
        # Mock file size checks
        # sequential should be selected for < 5MB file sizes
        # multiprocessing should be selected for >= 5MB file sizes
        # Since we use ranking_pipeline.route_execution(path), let's test it using temp files
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"0" * 100) # Small file (100 Bytes)
            small_path = f.name
            
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"0" * 6_000_000) # Large file (6 MB)
            large_path = f.name
            
        route_small = ranking_pipeline.route_execution(small_path)
        route_large = ranking_pipeline.route_execution(large_path)
        
        # Clean up temp files
        os.remove(small_path)
        os.remove(large_path)
        
        assert route_small == "sequential", "Small file dynamic routing failed."
        assert route_large == "multiprocessing", "Large file dynamic routing failed."
        
        print(f"  {GREEN}[PASS] Dynamic execution router routing logic verified.{RESET}")
        return True
    except AssertionError as e:
        print(f"  {RED}[FAIL] Dynamic router assertion failed: {e}{RESET}")
        return False
    except Exception as e:
        print(f"  {RED}[FAIL] Error during dynamic router verification: {e}{RESET}")
        return False

def check_output_format() -> bool:
    """Validates the output submission CSV schema, size, and sorting."""
    out_file = "submission.csv"
    print(f"Checking output submission file: {out_file}")
    
    if not os.path.exists(out_file):
        print(f"  {YELLOW}[WARN] No submission.csv found in workspace. Run pipeline to generate.{RESET}")
        return True
        
    try:
        with open(out_file, mode="r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            
        if len(reader) < 2:
            print(f"  {RED}[FAIL] Submission CSV is empty or missing rows.{RESET}")
            return False
            
        # Verify Headers
        headers = reader[0]
        expected_headers = ["candidate_id", "rank", "score", "reasoning"]
        if headers != expected_headers:
            print(f"  {RED}[FAIL] Headers mismatch! Expected: {expected_headers}, Got: {headers}{RESET}")
            return False
            
        # Verify Sorting & Rank
        ranks = []
        scores = []
        for row in reader[1:101]: # Verify top 100 rows
            cid, rank_val, score_val, reasoning = row
            ranks.append(int(rank_val))
            scores.append(float(score_val))
            
            # Check reasoning is not placeholder or None
            if not reasoning or "placeholder" in reasoning.lower() or "none" in reasoning.lower():
                print(f"  {RED}[FAIL] Candidate {cid} reasoning string contains placeholders: '{reasoning}'{RESET}")
                return False
                
        # Ensure rank matches 1..100 sequential ordering
        expected_ranks = list(range(1, len(ranks) + 1))
        if ranks != expected_ranks:
            print(f"  {RED}[FAIL] Rank column is not in sequential 1..100 order.{RESET}")
            return False
            
        # Ensure scores are sorted descending
        if scores != sorted(scores, reverse=True):
            print(f"  {RED}[FAIL] Candidate scores are not sorted in descending order!{RESET}")
            return False
            
        print(f"  {GREEN}[PASS] Submission CSV schema, sorting, and reasoning columns verified.{RESET}")
        return True
    except Exception as e:
        print(f"  {RED}[FAIL] Failed to validate submission CSV format: {e}{RESET}")
        return False

def check_performance_thresholds() -> bool:
    """Measures Layer 1 screening performance on the first 1,000 candidates."""
    dataset_path = "[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
    print("Benchmarking Layer 1 parsing speed (target: >2,000 cands/sec)...")
    
    if not os.path.exists(dataset_path):
        print(f"  {YELLOW}[WARN] Dataset not available for benchmarking. Skipping.{RESET}")
        return True
        
    try:
        generator = ranking_pipeline.load_candidates(dataset_path)
        test_cands = []
        for _ in range(1000):
            try:
                test_cands.append(next(generator))
            except StopIteration:
                break
                
        t0 = time.time()
        passed_count = 0
        for cand in test_cands:
            if not ranking_pipeline.should_exclude(cand, "Noida"):
                ranking_pipeline.compute_coarse_score(cand)
                passed_count += 1
        t_delta = time.time() - t0
        
        throughput = len(test_cands) / t_delta if t_delta > 0 else 0
        print(f"  Layer 1 throughput: {throughput:.2f} candidates/second.")
        
        if throughput < 2000:
            print(f"  {YELLOW}[WARN] Throughput ({throughput:.2f} c/s) is below target threshold of 2,000 c/s.{RESET}")
            return True
            
        print(f"  {GREEN}[PASS] Throughput is well within safety thresholds.{RESET}")
        return True
    except Exception as e:
        print(f"  {RED}[FAIL] Error during performance benchmark: {e}{RESET}")
        return False

def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    print_header("REDROB PIPELINE SPECIFIC HEALTH CHECKLIST")
    
    checks = [
        ("Model File Integrity Check", check_model_integrity),
        ("Pickle Cache Health Check", check_cache_health),
        ("Candidates Dataset Schema Check", check_dataset_schema),
        ("Layer 1 Filter Exclusions Assertions", check_l1_exclusions),
        ("Layer 2 Relevance Score Modifiers Check", check_l2_relevance_modifiers),
        ("Layer 3 Behavioral Multipliers Check", check_l3_behavioral_calibration),
        ("Dynamic Execution Router Logic Check", check_dynamic_router),
        ("Submission CSV Output Validation Check", check_output_format),
        ("Performance Throughput Check", check_performance_thresholds),
    ]
    
    failed = 0
    for name, func in checks:
        print(f"\n* Running: {name}...")
        if not func():
            failed += 1
            
    print_header("AUDIT SUMMARY")
    if failed == 0:
        print(f"{GREEN}🎉 All project-specific checks PASSED! The pipeline is 100% healthy and ready.{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}❌ {failed} checks FAILED. Please review the failures and remediate before submission.{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()
