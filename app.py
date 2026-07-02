import streamlit as st
import pandas as pd
import json
import os
import tempfile
import matplotlib.pyplot as plt

# Import the ranking pipeline logic
import ranking_pipeline

# Configure page settings
st.set_page_config(
    page_title="Redrob Ranker — Sandbox",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    h1 {
        color: #38BDF8;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #0284C7;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0369A1;
        box-shadow: 0px 4px 15px rgba(2, 132, 199, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Redrob Intelligent Candidate Ranker")
st.caption("Hosted Sandbox Sandbox Environment for Coarse & Behavioral Candidate Matching")

st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("""
This sandbox allows you to run the ranking pipeline on a small sample of candidates (up to 150) to verify reproducibility.
""")

# Sample candidates file location helper
default_sample_path = "./[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json"
sample_exists = os.path.exists(default_sample_path)

input_method = st.sidebar.radio(
    "Choose Input Method:",
    ["Use Default Hackathon Sample (50 Candidates)", "Upload Custom JSON/JSONL"] if sample_exists else ["Upload Custom JSON/JSONL"]
)

# Load candidate data
candidates_data = []
if input_method == "Use Default Hackathon Sample (50 Candidates)":
    st.info("Using preloaded 50 candidates from `sample_candidates.json`")
    candidates_data = list(ranking_pipeline.load_candidates(default_sample_path))
else:
    uploaded_file = st.sidebar.file_uploader("Upload JSON/JSONL Candidates", type=["json", "jsonl"])
    if uploaded_file is not None:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        candidates_data = list(ranking_pipeline.load_candidates(tmp_path))
        os.unlink(tmp_path)
        st.success(f"Successfully uploaded {len(candidates_data)} candidates!")

if len(candidates_data) > 0:
    st.write(f"### Loaded `{len(candidates_data)}` candidates for processing")
    
    if st.button("🚀 Run Ranking Pipeline"):
        with st.spinner("Processing layers (Filtering, Embeddings, Behavioral Trust score)..."):
            # Ensure model is ready
            if not os.path.exists(ranking_pipeline.MODEL_PATH):
                st.warning("Model directory './local_model' not found. Fetching model weights...")
                import download_model
                download_model.download_model()
            
            # Execute pipeline logic on the list
            eligible_candidates = []
            for cand in candidates_data:
                if not ranking_pipeline.should_exclude(cand, "Noida"):
                    coarse_score = ranking_pipeline.compute_coarse_score(cand)
                    eligible_candidates.append((coarse_score, cand.get("candidate_id", ""), cand))
            
            st.write(f"**Layer 1 Filter Survivors:** `{len(eligible_candidates)}` candidates.")
            
            if eligible_candidates:
                # Limit top survivors to score
                top_survivors = [(cand, coarse) for coarse, _, cand in sorted(eligible_candidates, key=lambda x: x[0], reverse=True)[:1500]]
                texts = [ranking_pipeline.get_candidate_text(cand) for cand, _ in top_survivors]
                
                # Semantic similarity
                cos_scores = ranking_pipeline.compute_semantic_similarity(
                    ranking_pipeline.JD_TEXT, texts, ranking_pipeline.MODEL_PATH
                )
                if cos_scores is None:
                    cos_scores = ranking_pipeline.fallback_semantic_scores(texts, ranking_pipeline.JD_TEXT)
                
                # Behavioral and Output layers
                final_results = []
                for idx, (cand, _) in enumerate(top_survivors):
                    sim = float(cos_scores[idx])
                    rel_score = ranking_pipeline.compute_relevance_score(cand, sim)
                    multiplier = ranking_pipeline.compute_behavioral_multiplier(cand)
                    final_score = rel_score * multiplier
                    reason = ranking_pipeline.generate_reasoning(cand, final_score)
                    
                    final_results.append({
                        "candidate_id": cand.get("candidate_id"),
                        "score": round(final_score, 4),
                        "reasoning": reason,
                        "title": cand.get("profile", {}).get("current_title", "N/A"),
                        "experience": cand.get("profile", {}).get("years_of_experience", 0)
                    })
                
                # Sort and Rank
                final_results.sort(key=lambda x: x["score"], reverse=True)
                df_results = pd.DataFrame(final_results)
                df_results.insert(0, "rank", range(1, len(df_results) + 1))
                
                # Format download csv
                csv_data = df_results[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)
                
                # Metrics dashboard
                col1, col2, col3 = st.columns(3)
                col1.metric("Highest Score", f"{df_results['score'].max():.4f}")
                col2.metric("Average YOE", f"{df_results['experience'].mean():.1f} Years")
                col3.metric("Passed Candidates", len(df_results))
                
                # Display dataframe
                st.write("### 🏆 Top Ranked Candidates")
                st.dataframe(
                    df_results[["rank", "candidate_id", "score", "title", "experience", "reasoning"]],
                    use_container_width=True
                )
                
                st.download_button(
                    label="📥 Download Ranked CSV",
                    data=csv_data,
                    file_name="team_antigravity.csv",
                    mime="text/csv"
                )
            else:
                st.error("No candidates survived Layer 1 hard filters.")
else:
    st.warning("Please upload a file or use the default sample to run the pipeline.")
