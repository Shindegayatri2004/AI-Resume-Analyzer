from fastapi import FastAPI, UploadFile, File
import shutil
import os

from modules.resume_parser import extract_text_from_pdf
from modules.skill_extractor import extract_skills_using_nlp
from modules.career_recommender import recommend_career

app = FastAPI()

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return {"message": "AI Resume Analyzer API running"}

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)):
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(file_path)
    skills = extract_skills_using_nlp(text)
    career = recommend_career(skills)

    return {
        "skills": skills,
        "career_recommendation": career
    }