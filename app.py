"""
app.py
------
AI-Based Resume Analysis and Career Recommendation System
Main Streamlit application entry point.

Run with: streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from modules.resume_parser import extract_resume_text, extract_contact_info, estimate_experience_years
from modules.skill_extractor import extract_skills_using_nlp, extract_education_level, extract_resume_sections
from modules.career_recommender import load_skills_dataset, recommend_career, generate_feedback
from utils.text_cleaner import clean_resume_text

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #6C63FF;
        --secondary: #FF6584;
        --accent: #43E97B;
        --bg-card: #1E1E2E;
        --bg-dark: #13131F;
        --text-primary: #E0E0FF;
        --text-muted: #8888AA;
        --border: rgba(108, 99, 255, 0.25);
    }

    html, body, .stApp {
        background: linear-gradient(135deg, #0D0D1A 0%, #13131F 50%, #0A0A18 100%);
        color: var(--text-primary);
        font-family: 'Space Grotesk', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF 0%, #FF6584 50%, #43E97B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: var(--text-muted);
        font-size: 1.05rem;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(255,101,132,0.06));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    .skill-chip {
        display: inline-block;
        background: linear-gradient(135deg, rgba(108,99,255,0.25), rgba(67,233,123,0.12));
        border: 1px solid rgba(108,99,255,0.4);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        margin: 0.2rem;
        font-size: 0.82rem;
        font-weight: 500;
        color: #C0B8FF;
    }

    .job-card {
        background: linear-gradient(135deg, rgba(30,30,46,0.9), rgba(19,19,31,0.9));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin: 0.6rem 0;
        position: relative;
        overflow: hidden;
    }
    .job-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #6C63FF, #43E97B);
        border-radius: 4px 0 0 4px;
    }
    .job-card .rank-badge {
        position: absolute;
        top: 1rem; right: 1rem;
        background: rgba(108,99,255,0.2);
        border: 1px solid rgba(108,99,255,0.4);
        border-radius: 50px;
        padding: 0.2rem 0.6rem;
        font-size: 0.72rem;
        color: #A89CFF;
        font-family: 'JetBrains Mono', monospace;
    }
    .job-card h3 {
        font-size: 1.15rem;
        font-weight: 600;
        color: #D0CCFF;
        margin: 0 0 0.3rem;
    }
    .job-card .cat-tag {
        font-size: 0.72rem;
        color: #43E97B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.7rem;
    }

    .confidence-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        height: 10px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }

    .feedback-item {
        background: rgba(108,99,255,0.08);
        border-left: 3px solid #6C63FF;
        border-radius: 0 10px 10px 0;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #C8C4E8;
    }

    .missing-skill-chip {
        display: inline-block;
        background: rgba(255,101,132,0.12);
        border: 1px solid rgba(255,101,132,0.35);
        border-radius: 20px;
        padding: 0.2rem 0.65rem;
        margin: 0.2rem;
        font-size: 0.79rem;
        font-weight: 500;
        color: #FF9EAF;
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #D0CCFF;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border);
    }

    /* Override Streamlit defaults */
    .stFileUploader > div { border: 2px dashed rgba(108,99,255,0.4) !important; background: rgba(108,99,255,0.05) !important; border-radius: 12px !important; }
    div[data-testid="stExpander"] { background: rgba(30,30,46,0.7) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
    .stAlert { border-radius: 12px !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: rgba(13,13,26,0.95) !important; border-right: 1px solid var(--border) !important; }
    section[data-testid="stSidebar"] .stMarkdown p { color: var(--text-muted); font-size: 0.88rem; }

    /* Tabs */
    .stTabs [role="tab"] { font-family: 'Space Grotesk', sans-serif; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def cached_load_dataset():
    """Cache the skills dataset to avoid repeated disk reads."""
    return load_skills_dataset()


def score_color(score: float) -> str:
    """Return a CSS hex color based on confidence score."""
    if score >= 70:
        return "#43E97B"   # green
    elif score >= 40:
        return "#FFD166"   # amber
    else:
        return "#FF6584"   # red


def render_skill_chips(skills, chip_class="skill-chip"):
    """Render a list of skills as HTML chips."""
    chips = "".join(f'<span class="{chip_class}">{s}</span>' for s in skills)
    st.markdown(chips, unsafe_allow_html=True)


def render_confidence_bar(score: float):
    """Render a custom HTML confidence progress bar."""
    color = score_color(score)
    st.markdown(f"""
    <div class="confidence-bar-bg">
        <div class="confidence-bar-fill" style="width:{score}%; background: linear-gradient(90deg, {color}88, {color});"></div>
    </div>
    <div style="font-size:0.78rem; color:{color}; font-weight:600; font-family:'JetBrains Mono',monospace;">{score}% match</div>
    """, unsafe_allow_html=True)


def build_skills_bar_chart(recommendations):
    """Create a Plotly horizontal bar chart for top role matches."""
    roles = [r["job_role"] for r in recommendations]
    scores = [r["confidence_pct"] for r in recommendations]
    colors = [score_color(s) for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=roles,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale=[[0, "#FF6584"], [0.5, "#FFD166"], [1, "#43E97B"]],
            cmin=0, cmax=100,
        ),
        text=[f"{s}%" for s in scores],
        textposition="outside",
        textfont=dict(color="#C0B8FF", size=12, family="JetBrains Mono"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#C0B8FF", family="Space Grotesk"),
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 120]),
        yaxis=dict(showgrid=False, tickfont=dict(size=13)),
        margin=dict(l=0, r=60, t=20, b=10),
        height=max(280, len(roles) * 52),
    )
    return fig


def build_radar_chart(skill_profile: dict):
    """Create a Plotly radar chart for skill category distribution."""
    categories = list(skill_profile.keys())
    values = list(skill_profile.values())

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        line=dict(color="#6C63FF", width=2),
        fillcolor="rgba(108,99,255,0.2)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#888AAA", size=9), gridcolor="rgba(255,255,255,0.07)"),
            angularaxis=dict(tickfont=dict(color="#C0B8FF", size=11), gridcolor="rgba(255,255,255,0.07)"),
        ),
        font=dict(color="#C0B8FF", family="Space Grotesk"),
        margin=dict(l=60, r=60, t=30, b=30),
        height=340,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 AI Resume Analyzer")
    st.markdown("---")
    st.markdown("""
    **How it works**

    1. Upload your resume (PDF or DOCX)
    2. AI extracts your skills using NLP
    3. Skills are matched against 20+ career paths
    4. Get personalized recommendations & gap analysis

    ---
    **Supported formats**
    - 📄 PDF (.pdf)
    - 📝 Word (.docx)

    ---
    **Version:** 1.0.0
    """)
    st.markdown("---")
    show_raw = st.checkbox("Show extracted raw text", value=False)
    top_n = st.slider("Number of career recommendations", min_value=3, max_value=10, value=5)


# ── Main Header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧠 AI Resume Analyzer</h1>
    <p>Upload your resume — get instant skill extraction, career matching & personalized recommendations</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Upload Section ────────────────────────────────────────────────────────────

col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("#### 📤 Upload Your Resume")
    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )

with col_info:
    st.markdown("#### 💡 Tips for Best Results")
    st.markdown("""
    <div style='font-size:0.85rem; color:#888AAA; line-height:1.8;'>
    ✅ Include a <b>Skills</b> section<br>
    ✅ List frameworks & tools explicitly<br>
    ✅ Use standard section headings<br>
    ✅ PDF format gives best parsing<br>
    </div>
    """, unsafe_allow_html=True)

# ── Analysis Pipeline ─────────────────────────────────────────────────────────

if uploaded_file is not None:
    with st.spinner("🔍 Analyzing your resume..."):

        # Step 1: Text extraction
        try:
            raw_text = extract_resume_text(uploaded_file)
        except ValueError as e:
            st.error(f"❌ {e}")
            st.stop()

        if not raw_text or len(raw_text.strip()) < 50:
            st.error("❌ Could not extract sufficient text from the resume. Please ensure the file is not scanned/image-based.")
            st.stop()

        # Step 2: Text cleaning
        clean_text = clean_resume_text(raw_text)

        # Step 3: Contact & metadata extraction
        contact_info = extract_contact_info(raw_text)
        experience_years = estimate_experience_years(raw_text)
        education_level = extract_education_level(raw_text)

        # Step 4: Skill extraction
        skills = extract_skills_using_nlp(clean_text)

        # Step 5: Career recommendations
        df = cached_load_dataset()
        report = recommend_career(skills, df, top_n=top_n)
        recommendations = report["top_recommendations"]
        primary = report["primary_recommendation"]
        skill_profile = report["skill_profile"]
        readiness = report["overall_readiness"]

        # Step 6: Feedback generation
        feedback_items = generate_feedback(skills, primary)

    st.success("✅ Resume analyzed successfully!")

    # ── Optional: Show raw extracted text ──────────────────────────────────────
    if show_raw:
        with st.expander("📄 Extracted Raw Text", expanded=False):
            st.text_area("", raw_text, height=200, label_visibility="collapsed")

    st.markdown("---")

    # ── Summary Metrics ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Resume Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Skills Detected</div>
            <div class="value">{len(skills)}</div>
            <div class="sub">from resume text</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        top_score = primary["confidence_pct"] if primary else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Best Match Score</div>
            <div class="value" style="color:{score_color(top_score)}">{top_score}%</div>
            <div class="sub">{primary['job_role'] if primary else 'N/A'}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Education Level</div>
            <div class="value" style="font-size:1rem; padding-top:0.4rem">{education_level}</div>
            <div class="sub">&nbsp;</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Career Readiness</div>
            <div class="value" style="font-size:1rem; padding-top:0.4rem">{readiness}</div>
            <div class="sub">&nbsp;</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Main Content Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Career Matches", "🛠️ Skills Analysis", "📈 Visualizations", "💬 Feedback"])

    # ── TAB 1: Career Matches ──────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">🎯 Recommended Career Paths</div>', unsafe_allow_html=True)

        if not recommendations:
            st.warning("No career matches found. Please ensure your resume includes identifiable skills.")
        else:
            for i, rec in enumerate(recommendations):
                rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"][i]
                color = score_color(rec["confidence_pct"])

                st.markdown(f"""
                <div class="job-card">
                    <div class="rank-badge">{rank_emoji} #{i+1}</div>
                    <h3>{rec['job_role']}</h3>
                    <div class="cat-tag">📂 {rec['category']}</div>
                """, unsafe_allow_html=True)

                render_confidence_bar(rec["confidence_pct"])

                col_m, col_miss = st.columns(2)
                with col_m:
                    st.markdown(f"**✅ Matched Skills** ({rec['match_count']}/{rec['total_required']})")
                    if rec["matched_skills"]:
                        render_skill_chips(rec["matched_skills"])
                    else:
                        st.caption("None matched")
                with col_miss:
                    st.markdown(f"**❌ Missing Skills** ({len(rec['missing_skills'])})")
                    if rec["missing_skills"]:
                        render_skill_chips(rec["missing_skills"][:8], chip_class="missing-skill-chip")
                    else:
                        st.caption("All skills matched! 🎉")

                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── TAB 2: Skills Analysis ─────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">🛠️ Detected Skills</div>', unsafe_allow_html=True)

        if not skills:
            st.warning("No skills detected. Consider adding a dedicated 'Technical Skills' section to your resume.")
        else:
            st.markdown(f"**{len(skills)} skills** extracted from your resume:")
            render_skill_chips(skills)

        st.markdown("---")

        # Contact info
        st.markdown('<div class="section-header">📇 Contact Information</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"📧 **Email:** {contact_info.get('email') or 'Not found'}")
            st.markdown(f"📞 **Phone:** {contact_info.get('phone') or 'Not found'}")
        with c2:
            st.markdown(f"💼 **LinkedIn:** {contact_info.get('linkedin') or 'Not found'}")
            st.markdown(f"🐙 **GitHub:** {contact_info.get('github') or 'Not found'}")

        # Top missing skills across all recommendations
        st.markdown("---")
        st.markdown('<div class="section-header">🔍 Skills Gap Overview</div>', unsafe_allow_html=True)

        # Aggregate missing skills across top 3 recommendations
        from collections import Counter
        all_missing = []
        for rec in recommendations[:3]:
            all_missing.extend(rec["missing_skills"])
        if all_missing:
            missing_counts = Counter(all_missing).most_common(15)
            st.markdown("Most frequently required skills you're missing across top 3 roles:")
            missing_df = pd.DataFrame(missing_counts, columns=["Skill", "Frequency"])
            fig_missing = px.bar(
                missing_df, x="Frequency", y="Skill", orientation="h",
                color="Frequency", color_continuous_scale=["#FF6584", "#FFD166"],
            )
            fig_missing.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#C0B8FF", family="Space Grotesk"),
                showlegend=False, coloraxis_showscale=False,
                margin=dict(l=0, r=20, t=10, b=10),
                height=max(220, len(missing_counts) * 38),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("🎉 No major skill gaps found across top recommendations!")

    # ── TAB 3: Visualizations ──────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">📈 Career Match Scores</div>', unsafe_allow_html=True)
        fig_bar = build_skills_bar_chart(recommendations)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        col_r, col_s = st.columns(2)
        with col_r:
            st.markdown('<div class="section-header">🕸️ Skill Category Profile</div>', unsafe_allow_html=True)
            if skill_profile:
                fig_radar = build_radar_chart(skill_profile)
                st.plotly_chart(fig_radar, use_container_width=True)

        with col_s:
            st.markdown('<div class="section-header">🥧 Match Distribution</div>', unsafe_allow_html=True)
            if primary:
                labels = ["Matched Skills", "Missing Skills"]
                values = [primary["match_count"], len(primary["missing_skills"])]
                fig_pie = go.Figure(go.Pie(
                    labels=labels, values=values,
                    hole=0.55,
                    marker=dict(colors=["#43E97B", "#FF6584"]),
                    textfont=dict(color="#fff", size=13, family="Space Grotesk"),
                ))
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#C0B8FF", family="Space Grotesk"),
                    legend=dict(font=dict(color="#C0B8FF")),
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=300,
                    annotations=[dict(
                        text=f"{primary['confidence_pct']}%",
                        x=0.5, y=0.5, font_size=22,
                        showarrow=False, font=dict(color="#D0CCFF", family="JetBrains Mono"),
                    )],
                )
                st.plotly_chart(fig_pie, use_container_width=True)

    # ── TAB 4: Feedback ────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">💬 Personalized Feedback & Recommendations</div>', unsafe_allow_html=True)
        for item in feedback_items:
            st.markdown(f'<div class="feedback-item">{item}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-header">🎓 Suggested Learning Resources</div>', unsafe_allow_html=True)

        if primary and primary["missing_skills"]:
            resources = {
                "Coursera": "https://www.coursera.org",
                "edX": "https://www.edx.org",
                "Udemy": "https://www.udemy.com",
                "Kaggle Learn": "https://www.kaggle.com/learn",
                "fast.ai": "https://www.fast.ai",
                "freeCodeCamp": "https://www.freecodecamp.org",
            }
            st.markdown("Explore these platforms to fill your skill gaps:")
            rc1, rc2, rc3 = st.columns(3)
            cols = [rc1, rc2, rc3]
            for idx, (name, url) in enumerate(resources.items()):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="metric-card" style="padding:0.8rem 1rem; text-align:center;">
                        <div style="font-size:0.9rem; font-weight:600; color:#A89CFF;">{name}</div>
                        <div style="font-size:0.75rem; margin-top:0.3rem;">
                            <a href="{url}" target="_blank" style="color:#6C63FF;">Visit →</a>
                        </div>
                    </div>""", unsafe_allow_html=True)

else:
    # ── Landing state (no file uploaded) ──────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #666688;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📋</div>
        <div style="font-size: 1.2rem; font-weight: 600; color: #9988CC; margin-bottom: 0.5rem;">
            Upload your resume to get started
        </div>
        <div style="font-size: 0.9rem; max-width: 500px; margin: 0 auto; line-height: 1.7; color: #5555AA;">
            The AI will extract your skills, compare them against 20+ job roles,
            and provide a personalized career roadmap with actionable improvements.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature preview cards
    st.markdown("---")
    st.markdown("#### ✨ What you'll get")
    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🔍", "Skill Extraction", "NLP-powered identification of 200+ technical & soft skills"),
        ("🎯", "Career Matching", "Confidence-scored match against 20+ job roles"),
        ("📊", "Visual Dashboard", "Charts and radar graphs of your skill profile"),
        ("💡", "Gap Analysis", "Prioritized list of skills to learn next"),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center; min-height:140px;">
                <div style="font-size:1.8rem; margin-bottom:0.5rem">{icon}</div>
                <div style="font-weight:600; color:#C0B8FF; font-size:0.95rem">{title}</div>
                <div style="color:#666688; font-size:0.8rem; margin-top:0.4rem; line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)
