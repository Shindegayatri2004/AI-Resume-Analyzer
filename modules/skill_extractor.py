"""
skill_extractor.py
------------------
Uses NLP techniques to extract skills from resume text.
Combines keyword matching against a skills database with
optional spaCy NER for improved accuracy.
"""

import re
import string
from typing import List, Set


# Comprehensive master skill list used for matching
MASTER_SKILLS = [
    # Programming Languages
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "R", "Go",
    "Rust", "Swift", "Kotlin", "PHP", "Ruby", "Scala", "MATLAB", "Perl",
    "Bash", "Shell", "Solidity", "Dart", "Lua",

    # Web Technologies
    "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Express.js",
    "Django", "Flask", "FastAPI", "Spring Boot", "ASP.NET", "GraphQL",
    "REST API", "Webpack", "Redux", "Next.js", "Nuxt.js", "Tailwind",
    "Bootstrap", "jQuery", "SASS", "SCSS", "Web Security", "OAuth",
    "Authentication", "Responsive Design", "Accessibility",

    # Data Science & ML
    "Machine Learning", "Deep Learning", "Neural Networks", "Statistics",
    "Data Analysis", "Data Visualization", "Feature Engineering",
    "Model Deployment", "A/B Testing", "Bayesian Methods",
    "Reinforcement Learning", "Transfer Learning", "Ensemble Methods",
    "Dimensionality Reduction", "Clustering", "Classification", "Regression",
    "Time Series", "Anomaly Detection",

    # AI & NLP
    "NLP", "Natural Language Processing", "NLTK", "spaCy", "Transformers",
    "BERT", "GPT", "Hugging Face", "Text Mining", "Sentiment Analysis",
    "Named Entity Recognition", "Information Extraction", "Computer Vision",
    "OpenCV", "Image Processing", "Object Detection", "YOLO", "GAN",
    "CNN", "RNN", "LSTM", "Attention Mechanism",

    # ML Frameworks
    "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "XGBoost",
    "LightGBM", "CatBoost", "Pandas", "NumPy", "Matplotlib", "Seaborn",
    "Plotly", "SciPy", "OpenAI", "Langchain", "CUDA",

    # Databases
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Oracle", "SQLite",
    "Redis", "Elasticsearch", "Cassandra", "DynamoDB", "Firebase",
    "Database Design", "Data Modeling", "ETL", "Data Warehousing",

    # Cloud & DevOps
    "AWS", "Azure", "Google Cloud", "GCP", "Docker", "Kubernetes",
    "CI/CD", "Jenkins", "GitLab", "GitHub Actions", "Terraform", "Ansible",
    "Linux", "Unix", "Bash", "Shell", "Nginx", "Apache",
    "Microservices", "Serverless", "Cloud", "Monitoring", "Networking",

    # Tools & Platforms
    "Git", "GitHub", "GitLab", "Jira", "Confluence", "Slack", "Jupyter",
    "VS Code", "IntelliJ", "Postman", "Tableau", "Power BI",
    "Excel", "Google Analytics", "Figma", "Adobe XD", "Sketch",

    # Business & Soft Skills (for analysts/managers)
    "Agile", "Scrum", "Kanban", "Project Management", "Product Strategy",
    "Stakeholder Management", "Business Analysis", "User Research",
    "Wireframing", "Prototyping", "Roadmapping", "Reporting",
    "Data-Driven", "Communication", "Leadership",

    # Security
    "Cybersecurity", "Network Security", "Penetration Testing", "SIEM",
    "Firewalls", "Cryptography", "Vulnerability Assessment",
    "Incident Response", "Compliance", "Risk Management",

    # Mobile
    "Android", "iOS", "React Native", "Flutter", "Xamarin", "Firebase",
    "App Store", "Push Notifications",

    # Other Technical
    "MLOps", "Data Pipeline", "Spark", "Hadoop", "Kafka", "Airflow",
    "Dbt", "Looker", "SSRS", "DAX", "Power Query", "VLOOKUP",
    "Web Scraping", "API Development", "Design Patterns",
    "Object-Oriented Programming", "Functional Programming",
    "System Design", "Algorithms", "Data Structures", "Problem Solving",
    "Testing", "Unit Testing", "TDD", "Debugging", "Code Review",
    "Documentation", "3D Modeling", "Unity", "Unreal Engine",
    "Smart Contracts", "Blockchain", "Web3.js", "Ethereum",
]


def _normalize(text: str) -> str:
    """Lowercase and remove punctuation for comparison."""
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def _build_skill_lookup(skills: List[str]) -> dict:
    """Build a normalized → original mapping for fast lookup."""
    return {_normalize(s): s for s in skills}


def extract_skills_using_nlp(text: str, custom_skills: List[str] = None) -> List[str]:
    """
    Extract skills from resume text using keyword matching and optional NLP.

    Strategy:
    1. Tokenize the text into n-grams (1, 2, 3 words).
    2. Match against master skill list (case-insensitive).
    3. Optionally use spaCy for additional entity recognition.

    Args:
        text: Cleaned resume text.
        custom_skills: Additional skills to include in matching.

    Returns:
        Sorted list of unique matched skills.
    """
    all_skills = MASTER_SKILLS.copy()
    if custom_skills:
        all_skills.extend(custom_skills)

    skill_lookup = _build_skill_lookup(all_skills)
    found_skills: Set[str] = set()

    # --- Pass 1: Exact n-gram matching ---
    words = re.findall(r"[\w\+#\.]+", text)

    # Check 1-grams, 2-grams, 3-grams
    for n in range(1, 4):
        for i in range(len(words) - n + 1):
            candidate = " ".join(words[i:i + n])
            normalized = _normalize(candidate)
            if normalized in skill_lookup:
                found_skills.add(skill_lookup[normalized])

    # --- Pass 2: Regex-based matching for multi-word skills ---
    for skill in all_skills:
        # Escape special regex characters (like C++)
        pattern = re.escape(skill)
        if re.search(rf"\b{pattern}\b", text, re.IGNORECASE):
            found_skills.add(skill)

    # --- Pass 3: spaCy NER for PRODUCT/ORG/TECH entities (optional) ---
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = None

        if nlp:
            doc = nlp(text[:50000])  # Limit for performance
            for ent in doc.ents:
                if ent.label_ in ("PRODUCT", "ORG", "WORK_OF_ART"):
                    normalized = _normalize(ent.text)
                    if normalized in skill_lookup:
                        found_skills.add(skill_lookup[normalized])
    except ImportError:
        pass  # spaCy not installed — gracefully skip

    return sorted(found_skills)


def extract_education_level(text: str) -> str:
    """
    Detect highest education level mentioned in resume text.

    Args:
        text: Raw or cleaned resume text.

    Returns:
        Education level string.
    """
    text_lower = text.lower()

    if any(kw in text_lower for kw in ["ph.d", "phd", "doctorate", "doctoral"]):
        return "PhD / Doctorate"
    elif any(kw in text_lower for kw in ["master", "m.s.", "m.sc", "mba", "m.tech", "m.e."]):
        return "Master's Degree"
    elif any(kw in text_lower for kw in ["bachelor", "b.s.", "b.sc", "b.tech", "b.e.", "b.a."]):
        return "Bachelor's Degree"
    elif any(kw in text_lower for kw in ["associate", "diploma", "certificate"]):
        return "Associate / Diploma"
    else:
        return "Not Specified"


def extract_resume_sections(text: str) -> dict:
    """
    Identify common resume sections for better parsing context.

    Args:
        text: Raw resume text.

    Returns:
        Dict mapping section names to their content snippets.
    """
    section_headers = {
        "summary": ["summary", "objective", "profile", "about"],
        "experience": ["experience", "work history", "employment", "work experience"],
        "education": ["education", "academic", "qualification"],
        "skills": ["skills", "technical skills", "competencies", "expertise"],
        "projects": ["projects", "portfolio", "work samples"],
        "certifications": ["certifications", "certificates", "courses", "training"],
    }

    sections = {}
    lines = text.split("\n")
    current_section = "other"
    sections[current_section] = []

    for line in lines:
        line_lower = line.lower().strip()
        matched_section = None
        for section, keywords in section_headers.items():
            if any(kw in line_lower for kw in keywords) and len(line_lower) < 50:
                matched_section = section
                break
        if matched_section:
            current_section = matched_section
            sections[current_section] = []
        else:
            sections.setdefault(current_section, []).append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items()}
