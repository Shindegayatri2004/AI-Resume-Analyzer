"""
text_cleaner.py
---------------
Utility functions for normalizing and cleaning raw text
extracted from resumes before NLP processing.
"""

import re
import unicodedata
from typing import List


def clean_resume_text(text: str) -> str:
    """
    Full cleaning pipeline for raw resume text.

    Steps:
        1. Normalize Unicode characters (accents, special chars).
        2. Remove non-printable / control characters.
        3. Collapse excessive whitespace and blank lines.
        4. Remove common resume noise (page numbers, watermarks, etc.).
        5. Preserve technical skill terms (e.g., C++, .NET, Node.js).

    Args:
        text: Raw text extracted from PDF or DOCX.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Step 1: Normalize Unicode (NFKD decomposes accented chars)
    text = unicodedata.normalize("NFKD", text)
    # Re-encode to ASCII, ignoring bytes that can't be encoded
    text = text.encode("ascii", "ignore").decode("ascii")

    # Step 2: Remove non-printable / control characters except whitespace
    text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)

    # Step 3: Normalize bullet points and list symbols
    text = re.sub(r"[•●◦▸▹►▻‣⁃∙·]", "-", text)

    # Step 4: Remove URLs (but keep domain keywords like github.com/username)
    text = re.sub(r"http[s]?://\S+", "", text)

    # Step 5: Remove email addresses (we capture them separately)
    text = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "", text)

    # Step 6: Remove standalone numbers (page numbers, phone formatting noise)
    text = re.sub(r"(?<!\w)\d{1,2}(?!\w)", " ", text)

    # Step 7: Normalize whitespace — collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Step 8: Collapse multiple blank lines into at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 9: Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into individual words, keeping technical terms intact.

    Preserves tokens like: C++, .NET, Node.js, scikit-learn, co-worker.

    Args:
        text: Cleaned resume text.

    Returns:
        List of string tokens.
    """
    # Match word-like tokens including dots, plus signs, hyphens as part of names
    tokens = re.findall(r"[A-Za-z][\w\+#\.\-]*", text)
    return [t for t in tokens if len(t) > 1]


def remove_stopwords(tokens: List[str], extra_stopwords: List[str] = None) -> List[str]:
    """
    Filter out common English stopwords and resume filler words.

    Args:
        tokens: List of word tokens.
        extra_stopwords: Additional domain-specific words to exclude.

    Returns:
        Filtered list of meaningful tokens.
    """
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "through", "during",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
        "this", "that", "these", "those", "who", "which", "what", "when", "where",
        # Resume-specific filler
        "responsible", "duties", "role", "position", "worked", "working",
        "team", "company", "organization", "department", "including",
        "skills", "skill", "experience", "knowledge", "ability", "proficient",
        "strong", "excellent", "good", "great", "various", "multiple", "also",
        "as", "well", "such", "etc", "eg", "ie", "refer",
    }
    if extra_stopwords:
        STOPWORDS.update(w.lower() for w in extra_stopwords)

    return [t for t in tokens if t.lower() not in STOPWORDS]


def extract_noun_phrases(text: str) -> List[str]:
    """
    Extract simple noun phrases (adjective + noun combos) using regex.
    Falls back to noun-only extraction if NLP is unavailable.

    Args:
        text: Cleaned resume text.

    Returns:
        List of noun phrase strings.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text[:30000])
            return [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4]
        except OSError:
            pass
    except ImportError:
        pass

    # Fallback: simple capitalized phrase extraction
    pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
    return re.findall(pattern, text)


def normalize_skill_name(skill: str) -> str:
    """
    Normalize a skill name for display (title case with exceptions).

    Args:
        skill: Raw skill string.

    Returns:
        Normalized skill string.
    """
    # List of skills that should stay uppercase
    UPPERCASE_SKILLS = {
        "sql", "html", "css", "api", "nlp", "ml", "ai", "gpu", "cpu",
        "aws", "gcp", "sap", "erp", "crm", "etl", "sdk", "ide", "orm",
        "tdd", "bdd", "oop", "seo", "ux", "ui", "ci", "cd", "vcs", "ssrs",
        "dax", "bi", "php", "r", "c", "qa"
    }
    lower = skill.strip().lower()
    if lower in UPPERCASE_SKILLS:
        return skill.strip().upper()
    return skill.strip()
