"""
career_recommender.py
---------------------
Matches extracted resume skills against job role requirements
from the skills dataset and recommends the best career paths.
"""

import os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


def load_skills_dataset(csv_path: str = None) -> pd.DataFrame:
    """
    Load the job roles and skills dataset from CSV.

    Args:
        csv_path: Path to skills_dataset.csv. Defaults to file in project root.

    Returns:
        DataFrame with columns: Job Role, Skills (list), Category.
    """
    if csv_path is None:
        # Resolve relative to this file's location
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "skills_dataset.csv")

    try:
        df = pd.read_csv(csv_path)
        # Parse semicolon-separated skills into Python lists
        df["Skills_List"] = df["Skills"].apply(
            lambda x: [s.strip() for s in str(x).split(";") if s.strip()]
        )
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Skills dataset not found at: {csv_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load skills dataset: {e}")


def compute_match_score(resume_skills: List[str], job_skills: List[str]) -> float:
    """
    Compute a Jaccard-like similarity score between resume skills and job requirements.

    Uses case-insensitive matching to avoid penalising capitalisation differences.

    Args:
        resume_skills: List of skills found in the resume.
        job_skills: List of skills required for a job role.

    Returns:
        Float score between 0.0 and 1.0.
    """
    if not job_skills:
        return 0.0

    resume_lower = {s.lower() for s in resume_skills}
    job_lower = {s.lower() for s in job_skills}

    matched = resume_lower & job_lower
    score = len(matched) / len(job_lower)
    return round(score, 4)


def get_matched_and_missing_skills(
    resume_skills: List[str], job_skills: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Separate skills into matched (present in resume) and missing (not in resume).

    Args:
        resume_skills: Skills found in resume.
        job_skills: Skills required for a specific job role.

    Returns:
        Tuple of (matched_skills, missing_skills).
    """
    resume_lower = {s.lower(): s for s in resume_skills}
    matched, missing = [], []
    for skill in job_skills:
        if skill.lower() in resume_lower:
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing


def match_resume_skills_with_job_roles(
    resume_skills: List[str],
    df: pd.DataFrame,
    top_n: int = 5,
) -> List[Dict]:
    """
    Score and rank all job roles against the resume skills.

    Args:
        resume_skills: Skills extracted from the resume.
        df: Skills dataset DataFrame.
        top_n: Number of top recommendations to return.

    Returns:
        List of dicts (sorted by score desc) with keys:
            - job_role, category, score, confidence_pct,
              matched_skills, missing_skills, total_required
    """
    results = []
    for _, row in df.iterrows():
        job_skills = row["Skills_List"]
        score = compute_match_score(resume_skills, job_skills)
        matched, missing = get_matched_and_missing_skills(resume_skills, job_skills)
        results.append({
            "job_role": row["Job Role"],
            "category": row.get("Category", "General"),
            "score": score,
            "confidence_pct": round(score * 100, 1),
            "matched_skills": matched,
            "missing_skills": missing,
            "total_required": len(job_skills),
            "match_count": len(matched),
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


def recommend_career(
    resume_skills: List[str],
    df: pd.DataFrame = None,
    top_n: int = 5,
) -> Dict:
    """
    Main entry point: produce a full career recommendation report.

    Args:
        resume_skills: Extracted skills list.
        df: Optional preloaded DataFrame (loaded from disk if None).
        top_n: Number of recommendations.

    Returns:
        Dict containing:
            - top_recommendations: list of ranked career matches
            - primary_recommendation: single best match
            - skill_profile: category distribution
            - overall_readiness: broad readiness label
    """
    if df is None:
        df = load_skills_dataset()

    recommendations = match_resume_skills_with_job_roles(resume_skills, df, top_n=top_n)

    # Compute a skill-category profile from the resume
    all_categories = df["Category"].unique().tolist()
    category_scores = {}
    for cat in all_categories:
        cat_df = df[df["Category"] == cat]
        cat_matches = match_resume_skills_with_job_roles(
            resume_skills, cat_df, top_n=len(cat_df)
        )
        if cat_matches:
            category_scores[cat] = round(
                float(np.mean([r["score"] for r in cat_matches])) * 100, 1
            )

    # Determine readiness tier
    top_score = recommendations[0]["confidence_pct"] if recommendations else 0
    if top_score >= 70:
        readiness = "🟢 Job-Ready"
    elif top_score >= 45:
        readiness = "🟡 Developing"
    else:
        readiness = "🔴 Foundational"

    return {
        "top_recommendations": recommendations,
        "primary_recommendation": recommendations[0] if recommendations else None,
        "skill_profile": category_scores,
        "overall_readiness": readiness,
        "total_skills_found": len(resume_skills),
    }


def generate_feedback(resume_skills: List[str], top_recommendation: Dict) -> List[str]:
    """
    Generate actionable resume improvement feedback.

    Args:
        resume_skills: Skills extracted from the resume.
        top_recommendation: The primary career recommendation dict.

    Returns:
        List of feedback string messages.
    """
    feedback = []
    n_skills = len(resume_skills)

    # General feedback based on skill count
    if n_skills < 5:
        feedback.append(
            "📝 Your resume has very few detectable skills. "
            "Consider adding a dedicated 'Technical Skills' section."
        )
    elif n_skills < 15:
        feedback.append(
            "📝 You have a moderate number of skills. "
            "Expanding your skill set and ensuring they appear explicitly will boost matching."
        )
    else:
        feedback.append(
            f"✅ Great — {n_skills} skills detected. "
            "Make sure these are highlighted prominently in your resume."
        )

    # Top role feedback
    if top_recommendation:
        role = top_recommendation["job_role"]
        score = top_recommendation["confidence_pct"]
        missing = top_recommendation["missing_skills"]

        if score >= 70:
            feedback.append(
                f"🎯 You are a strong candidate for **{role}** ({score}% match). "
                "Focus on portfolio projects to showcase these skills."
            )
        elif score >= 40:
            feedback.append(
                f"📈 You are a developing candidate for **{role}** ({score}% match). "
                "Upskilling in the missing areas will significantly improve your prospects."
            )
        else:
            feedback.append(
                f"🛠️ You have a foundational match with **{role}** ({score}%). "
                "Consider starting with online courses to fill the skill gaps."
            )

        if missing:
            top_missing = missing[:5]
            feedback.append(
                f"🔍 Priority skills to learn for **{role}**: "
                + ", ".join(f"**{s}**" for s in top_missing)
                + "."
            )

    # Generic best-practice tips
    feedback.append(
        "📌 Use action verbs and quantify achievements "
        "(e.g., 'Improved model accuracy by 12%') for stronger impact."
    )
    feedback.append(
        "🔗 Consider adding GitHub/LinkedIn links and showcasing 2–3 relevant projects."
    )

    return feedback
