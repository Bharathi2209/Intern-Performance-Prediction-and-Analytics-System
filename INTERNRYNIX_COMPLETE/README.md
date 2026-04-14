# InternRynix — Intern Performance Prediction System

## Sprint 4 — Model Deployment

Live URL: `https://internrynix.onrender.com`

---

## What This App Does

Predicts intern performance as **High / Medium / Low** and gives an exact **performance score** based on:
- Academic background (CGPA, backlogs, certifications)
- Work behavior (attendance, tasks, deadline adherence)
- Soft skills (communication, teamwork, initiative, adaptability, mentor rating)

---

## How to Run Locally

```bash
pip install -r requirements.txt
python train_save_model.py
python app.py
```

Open: `http://127.0.0.1:5000`

---

## API Endpoint

### POST /api/predict

```bash
curl -X POST https://internrynix.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cgpa": 8.5,
    "attendance_pct": 92,
    "tasks_completed": 30,
    "deadline_adherence": 88,
    "communication_score": 8,
    "teamwork_rating": 8,
    "initiative_score": 9,
    "adaptability_score": 7,
    "mentor_rating": 8,
    "backlogs": 0,
    "bugs_or_errors": 2,
    "certifications_count": 3,
    "internship_exp_months": 3,
    "overtime_hours": 5,
    "age": 22,
    "gender": "Male",
    "department": "Engineering",
    "college_tier": "Tier 2"
  }'
```

### Response

```json
{
  "status": "success",
  "performance_label": "High",
  "performance_score": 118.4,
  "confidence_pct": 94.2,
  "probabilities": {
    "Low": 1.2,
    "Medium": 4.6,
    "High": 94.2
  }
}
```

---

## Model Details

| Task           | Algorithm          | Metric     | Score  |
|----------------|--------------------|------------|--------|
| Classification | Gradient Boosting  | Accuracy   | 98.75% |
| Classification | Gradient Boosting  | F1 Macro   | 0.9874 |
| Regression     | Gradient Boosting  | R² Score   | 0.9929 |
| Regression     | Gradient Boosting  | MAE        | 2.16 pts |

---

## Sprint Summary

| Sprint | Focus | Output |
|--------|-------|--------|
| Sprint 1 | Data Foundation | 800 records, cleaned dataset |
| Sprint 2 | Model Development | RF vs GB, 93% accuracy |
| Sprint 3 | Optimization | 6 new features, 98.75% accuracy |
| Sprint 4 | Deployment | Live Flask app on Render |

---

## Project Structure

```
INTERNRYNIX/
├── data/
│   ├── data_raw.csv
│   └── data_cleaned.csv
├── model/
│   ├── model_clf.pkl
│   ├── model_reg.pkl
│   └── ...
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── batch_result.html
│   └── about.html
├── app.py
├── train_save_model.py
├── requirements.txt
├── Procfile
└── README.md
```
