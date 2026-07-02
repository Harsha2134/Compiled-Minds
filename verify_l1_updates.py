#!/usr/bin/env python3
"""
Verification Script for Layer 1 Upgrades:
- Location commuter hubs allowance (Pune, Mumbai, Hyderabad, Delhi NCR, etc.)
- Temporal future signup date honeypot check (July 2, 2026 baseline)
- Outsourcing filter trap bypass (allow product-minded engineers who currently work at IT services firms).
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ranking_pipeline

def run_verification():
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    print("=" * 60)
    print("      REDROB LAYER 1 UPGRADE VERIFICATION      ")
    print("=" * 60)

    # Base valid candidate template
    base_candidate = {
        "candidate_id": "cand_test",
        "profile": {
            "years_of_experience": 5,
            "location": "Noida, UP",
            "current_title": "Software Engineer",
            "headline": "Search Engineer",
            "summary": "Building search pipelines."
        },
        "redrob_signals": {
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
            "profile_completeness_score": 85,
            "preferred_work_mode": "onsite",
            "willing_to_relocate": False,
            "signup_date": "2026-05-10"
        },
        "career_history": [
            {"company": "Product Co", "title": "Developer", "duration_months": 24, "is_current": True}
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 24}
        ],
        "education": [
            {"end_year": 2021}
        ]
    }

    # -------------------------------------------------------------------------
    # TEST 1: Location Commuter Hubs Allowance
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: Location Commuter Hubs ---")
    locations_to_test = {
        "Pune, Maharashtra": False,      # Target city - should NOT exclude
        "Mumbai, Maharashtra": False,    # Commuter hub - should NOT exclude
        "Hyderabad, Telangana": False,   # Commuter hub - should NOT exclude
        "Gurgaon, Haryana": False,       # Delhi NCR - should NOT exclude
        "Bangalore, Karnataka": True     # Non-hub, onsite, unwilling to relocate - should EXCLUDE
    }

    for loc, expect_exclude in locations_to_test.items():
        cand = dict(base_candidate)
        cand["profile"] = dict(base_candidate["profile"])
        cand["profile"]["location"] = loc
        
        excluded = ranking_pipeline.should_exclude(cand)
        print(f"Location: {loc:<25} | Excluded: {excluded:<5} | Expected: {expect_exclude}")
        assert excluded == expect_exclude, f"Failed location check for {loc}"

    # -------------------------------------------------------------------------
    # TEST 2: Temporal Future Signup Date honeypot check
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: Future Signup Date Honeypot ---")
    dates_to_test = {
        "2026-06-15": False,  # Legit signup in June 2026 - should NOT exclude
        "2026-07-01": False,  # Legit signup yesterday - should NOT exclude
        "2026-07-02": False,  # Legit signup today - should NOT exclude
        "2026-07-05": True,   # Future date (after July 2, 2026) - should EXCLUDE
        "2028-01-01": True    # Far future date - should EXCLUDE
    }

    for s_date, expect_exclude in dates_to_test.items():
        cand = dict(base_candidate)
        cand["redrob_signals"] = dict(base_candidate["redrob_signals"])
        cand["redrob_signals"]["signup_date"] = s_date
        
        excluded = ranking_pipeline.should_exclude(cand)
        print(f"Signup Date: {s_date:<12} | Excluded: {excluded:<5} | Expected: {expect_exclude}")
        assert excluded == expect_exclude, f"Failed future date check for {s_date}"

    # -------------------------------------------------------------------------
    # TEST 3: Outsourcing Filter Trap Bypass
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: Outsourcing Filter Trap ---")
    
    # Candidate 3A: Only worked at IT services
    cand_services_only = dict(base_candidate)
    cand_services_only["career_history"] = [
        {"company": "TCS", "title": "Developer", "duration_months": 12},
        {"company": "Infosys", "title": "Engineer", "duration_months": 12}
    ]
    excluded_services = ranking_pipeline.should_exclude(cand_services_only)
    print(f"IT Services Only Career | Excluded: {excluded_services:<5} | Expected: True")
    assert excluded_services is True, "IT services only should be excluded"

    # Candidate 3B: Currently works at TCS but worked at Product Co previously
    cand_mixed_career = dict(base_candidate)
    cand_mixed_career["career_history"] = [
        {"company": "TCS", "title": "Developer", "duration_months": 12, "is_current": True},
        {"company": "Product Startup", "title": "ML Engineer", "duration_months": 24, "is_current": False}
    ]
    excluded_mixed = ranking_pipeline.should_exclude(cand_mixed_career)
    print(f"Mixed (TCS + Product Co) | Excluded: {excluded_mixed:<5} | Expected: False")
    assert excluded_mixed is False, "Mixed career containing product company should pass"

    print("\n" + "=" * 60)
    print("🎉 ALL LAYER 1 UPGRADE VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
