# 🧠 AI-Based Resume Analysis and Career Recommendation System

A full-stack AI application that analyzes resumes using Natural Language Processing (NLP)
to extract skills, match them against industry requirements, and recommend suitable career paths.

---

## 🌟 Features

| Feature | Description |
|---|---|
| 📤 Resume Upload | Supports PDF and DOCX formats |
| 🔍 Skill Extraction | NLP keyword matching + optional spaCy NER |
| 🎯 Career Matching | Confidence-scored recommendations for 20+ roles |
| 📊 Visual Dashboard | Bar charts, radar graphs, pie charts |
| 💡 Gap Analysis | Prioritized missing skills per role |
| 💬 Feedback | Actionable improvement suggestions |
| 📇 Contact Extraction | Email, phone, LinkedIn, GitHub detection |

---

## 🛠️ Technology Stack

- **Frontend/UI**: Streamlit
- **NLP**: spaCy, NLTK, regex-based extraction
- **ML/Data**: Scikit-learn, Pandas, NumPy
- **Visualization**: Plotly
- **PDF Parsing**: pdfplumber, PyPDF2
- **DOCX Parsing**: python-docx
- **Language**: Python 3.9+

---

## 📁 Project Structure

```
AI_Resume_Analyzer/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── skills_dataset.csv          # Job roles & required skills database
│
├── modules/
│   ├── __init__.py
│   ├── resume_parser.py        # PDF/DOCX text extraction
│   ├── skill_extractor.py      # NLP-based skill identification
│   └── career_recommender.py  # Career matching & scoring engine
│
├── utils/
│   ├── __init__.py
│   └── text_cleaner.py        # Text normalization utilities
│
└── sample_resumes/
    └── sample_resume.txt      # Sample resume for testing
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1: Clone or Download the Project

```bash
git clone https://github.com/yourusername/AI_Resume_Analyzer.git
cd AI_Resume_Analyzer
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download spaCy Language Model (Optional but Recommended)

```bash
python -m spacy download en_core_web_sm
```

> **Note:** If spaCy's language model is not installed, the system falls back to
> regex-based skill extraction — it will still work well!

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The application will open at **http://localhost:8501** in your browser.

---

## 📖 How to Use

1. Open the app in your browser
2. Click **"Browse files"** or drag & drop your resume (PDF or DOCX)
3. Wait a few seconds for analysis to complete
4. Explore the four tabs:
   - **Career Matches** – ranked job role recommendations with skill breakdowns
   - **Skills Analysis** – all detected skills + gap analysis chart
   - **Visualizations** – charts of match scores, skill profile radar, pie chart
   - **Feedback** – personalized improvement tips and learning resources

---

## 📊 Skills Dataset

The `skills_dataset.csv` contains 20+ job roles across categories:

| Category | Roles |
|---|---|
| Data & AI | Data Scientist, ML Engineer, Data Analyst, NLP Engineer, etc. |
| Web & Software | Web Developer, Frontend, Backend, Full Stack, Mobile |
| Infrastructure & Cloud | DevOps, Cloud Architect, Cybersecurity, DBA |
| Emerging Tech | Blockchain Developer, AR/VR Developer |
| Management & Design | Product Manager, UI/UX Designer |

You can extend this dataset by editing `skills_dataset.csv` and adding new rows.

---

## 🧩 Module Documentation

### `resume_parser.py`
- `extract_text_from_pdf(file)` – Extract text from PDF using pdfplumber
- `extract_text_from_docx(file)` – Extract text from DOCX using python-docx
- `extract_resume_text(uploaded_file)` – Auto-detect format and extract
- `extract_contact_info(text)` – Regex-based contact detail extraction
- `estimate_experience_years(text)` – Heuristic experience estimation

### `skill_extractor.py`
- `extract_skills_using_nlp(text)` – Multi-pass skill extraction (n-gram + NER)
- `extract_education_level(text)` – Detect highest education qualification
- `extract_resume_sections(text)` – Segment resume into standard sections

### `career_recommender.py`
- `load_skills_dataset(csv_path)` – Load and parse the skills CSV
- `compute_match_score(resume_skills, job_skills)` – Jaccard similarity score
- `match_resume_skills_with_job_roles(skills, df)` – Score all job roles
- `recommend_career(skills, df, top_n)` – Full recommendation report
- `generate_feedback(skills, top_recommendation)` – Actionable feedback

### `utils/text_cleaner.py`
- `clean_resume_text(text)` – Full cleaning pipeline (Unicode, noise removal)
- `tokenize_words(text)` – Tokenize preserving technical terms
- `remove_stopwords(tokens)` – Filter filler words
- `normalize_skill_name(skill)` – Standardize skill display names

---

## 🧪 Testing with Sample Resume

A sample resume is provided in `sample_resumes/sample_resume.txt`.
Convert it to PDF using any word processor to test the full pipeline.

---

## ⚙️ Configuration

- **Top N recommendations**: Adjustable via sidebar slider (3–10)
- **Skills dataset**: Edit `skills_dataset.csv` to add/modify roles
- **Skill list**: Extend `MASTER_SKILLS` in `skill_extractor.py`

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| PDF text is empty | File may be image-based (scanned). Use a text PDF. |
| spaCy model not found | Run `python -m spacy download en_core_web_sm` |
| Port already in use | Use `streamlit run app.py --server.port 8502` |
| No skills detected | Ensure resume has an explicit "Skills" section |

---

## 📄 License

MIT License — Free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [spaCy](https://spacy.io/) for NLP infrastructure
- [Streamlit](https://streamlit.io/) for the web framework
- [Plotly](https://plotly.com/) for interactive visualizations
- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF parsing
