#!/usr/bin/env python3
"""
Verification Script for Layer 3 (Behavioral Calibration Powerlayer Upgrade).
Validates that the class-based BehavioralCalibrationLayer works correctly,
distinguishes normal candidates from suspicious bots, and returns accurate
anomaly flags and trust scores.
"""

import sys
import os

# Ensure we can import modules from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ranking_pipeline

def run_verification():
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    print("=" * 60)
    print("      REDROB LAYER 3 BEHAVIORAL CALIBRATION VERIFICATION      ")
    print("=" * 60)
    calibrator = ranking_pipeline.BehavioralCalibrationLayer()

    # Case A: A legitimate, high-quality senior candidate
    good_candidate = {
        "id": "USR_001",
        "profile": {
            "years_of_experience": 7,
            "current_title": "Senior AI Engineer",
            "location": "Noida"
        },
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 48},
            {"name": "NLP", "proficiency": "expert", "duration_months": 36}
        ],
        "career_history": [
            {"company": "Google", "title": "AI Engineer", "duration_months": 48, "is_current": True}
        ],
        "education": [{"end_year": 2019}],
        "redrob_signals": {
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
            "profile_completeness_score": 95,
            "signup_date": "2024-03-15",
            "connection_count": 450,
            "recruiter_response_rate": 0.80,
            "avg_response_time_hours": 4.5,
            "applications_submitted_30d": 12,
            "interview_completion_rate": 0.90,
            "skill_assessment_scores": {"Python": 88, "NLP": 82},
            "github_activity_score": 65,
            "notice_period_days": 15,
            "saved_by_recruiters_30d": 8
        }
    }

    # Case B: An advanced bot script that is gaming the platform
    bot_candidate = {
        "id": "USR_BOT_99",
        "profile": {
            "years_of_experience": 1,
            "current_title": "Developer",
            "location": "Bangalore"
        },
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 2}
        ],
        "career_history": [
            {"company": "Unknown", "title": "Developer", "duration_months": 3, "is_current": True}
        ],
        "education": [{"end_year": 2025}],
        "redrob_signals": {
            "verified_email": False,  # Missing verification
            "verified_phone": True,
            "linkedin_connected": False, # Missing verification
            "profile_completeness_score": 100,
            "signup_date": "2026-05-25", # Signed up days ago (relative to 2026-06-01 baseline)
            "connection_count": 1200,    # Gained >1000 connections
            "endorsements_received": 150, # Gained >100 endorsements
            "recruiter_response_rate": 0.99, # Instant bot replies
            "avg_response_time_hours": 0.01, # Instant bot replies
            "applications_submitted_30d": 95, # Spamming applications
            "interview_completion_rate": 0.05, # Skipping interviews
            "skill_assessment_scores": {"Python": 100}, # Perfect score
            "github_activity_score": -1,  # No GitHub linked
            "notice_period_days": 100,
            "saved_by_recruiters_30d": 0,
            "search_appearance_30d": 50,
            "open_to_work_flag": True
        }
    }

    # Evaluate Good Candidate
    print("\n--- GOOD CANDIDATE EVALUATION ---")
    good_mult = ranking_pipeline.compute_behavioral_multiplier(good_candidate)
    trust_good, flags_good = calibrator.calculate_trust_score(good_candidate)
    print(f"Candidate ID:             {good_candidate['id']}")
    print(f"Behavioral Trust Score:   {trust_good:.2f}")
    print(f"Triggered Anomaly Flags:  {flags_good}")
    print(f"Final Score Multiplier:   {good_mult:.4f}")
    print(f"Manual Review Flagged:    {good_candidate.get('manual_verification_flag', False)}")
    
    # Assertions for Good Candidate
    assert trust_good == 1.0, "Good candidate should have 1.0 trust score."
    assert not flags_good, "Good candidate should not trigger any anomaly flags."
    assert good_candidate.get("manual_verification_flag", False) is False, "Good candidate should not require review."

    # Evaluate Bot Candidate
    print("\n--- BOT CANDIDATE EVALUATION ---")
    bot_mult = ranking_pipeline.compute_behavioral_multiplier(bot_candidate)
    trust_bot, flags_bot = calibrator.calculate_trust_score(bot_candidate)
    print(f"Candidate ID:             {bot_candidate['id']}")
    print(f"Behavioral Trust Score:   {trust_bot:.2f}")
    print(f"Triggered Anomaly Flags:  {flags_bot}")
    print(f"Final Score Multiplier:   {bot_mult:.4f}")
    print(f"Manual Review Flagged:    {bot_candidate.get('manual_verification_flag', False)}")
    print(f"Flagged Reason:           {bot_candidate.get('verification_reason', '')}")
    
    # Assertions for Bot Candidate
    assert trust_bot < 0.10, "Bot candidate trust score should be near 0."
    assert len(flags_bot) >= 4, "Bot candidate should trigger multiple anomalies."
    assert bot_candidate.get("manual_verification_flag", False) is True, "Bot candidate must be flagged for manual review."
    
    print("\n" + "=" * 60)
    print("🎉 VERIFICATION PASSED: Behavioral Powerlayer functions perfectly!")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
