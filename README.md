# Single-File AI Resume Screening System

A clean, self-contained AI Resume Screening & ATS Candidate Ranking System built with Python, Streamlit, and NLP libraries (`scikit-learn`, `pypdf`, `python-docx`, `plotly`).

---

## 📁 Project Structure

```
AI_Resume_Project/
├── 📄 app.py              # Complete Single-File Streamlit Application
├── 📜 requirements.txt    # Python Dependencies
├── 🙈 .gitignore          # Git Ignore Rules
└── 📖 README.md           # Documentation
```

---

## 🚀 Key Features

- **Multi-Resume Upload:** Ingest `.pdf`, `.docx`, and `.txt` candidate resume files.
- **Job Description Input:** Paste job requirements or upload job description files.
- **Text Parser & NLP Preprocessing:** Text extraction, lowercasing, normalization, and contact extraction.
- **Skill Extraction:** Keyword and taxonomy matching against technical & soft skill sets.
- **ATS Match Scoring:** Composite score calculated using **TF-IDF Vectorization** and **Cosine Similarity**.
- **Candidate Leaderboard:** Ranked list with High Fit (≥70%), Medium Fit (45-69%), and Low Fit (<45%) categorization.
- **Candidate Scorecards:** Visual matched skill pills (green) and missing skill pills (red).
- **Plotly Analytics Dashboard:** Donut chart, ATS score distribution histogram, top candidate comparison bar chart, and skill gap chart.
- **CSV Export:** Download candidate rankings as `.csv`.

---

## 🛠️ Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Application
```bash
streamlit run app.py
```
