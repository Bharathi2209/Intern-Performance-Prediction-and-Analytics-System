"""
train_save_model.py
Sprint 4 — Train and save the best models as .pkl files
Run this ONCE before starting app.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

os.makedirs('model', exist_ok=True)

print("=" * 55)
print("  SPRINT 4 — TRAINING & SAVING MODELS")
print("=" * 55)

df = pd.read_csv('data/data_cleaned.csv')

# Feature engineering (same as Sprint 3)
df['academic_index']    = round((df['cgpa']/10)*0.7 + (df['certifications_count']/8)*0.3, 4)
df['soft_skill_avg']    = round((df['communication_score']+df['teamwork_rating']+df['initiative_score']+df['adaptability_score']+df['mentor_rating'])/5, 4)
df['work_efficiency']   = round(df['tasks_completed']/(df['bugs_or_errors']+1), 4)
df['reliability_score'] = round((df['attendance_pct']/100)*(df['deadline_adherence']/100)*100, 4)
df['experience_score']  = round(df['internship_exp_months']*0.6 + df['certifications_count']*0.4, 4)
df['potential_index']   = round(df['academic_index']*30 + df['soft_skill_avg']*25 + df['work_efficiency']*20 + df['reliability_score']*15 + df['experience_score']*10, 4)

# Encode
le_gender = LabelEncoder()
le_dept   = LabelEncoder()
le_tier   = LabelEncoder()

df['gender']       = le_gender.fit_transform(df['gender'].astype(str))
df['department']   = le_dept.fit_transform(df['department'].astype(str))
df['college_tier'] = le_tier.fit_transform(df['college_tier'].astype(str))

label_map = {'Low':0,'Medium':1,'High':2}
df['performance_label'] = df['performance_label'].map(label_map)

drop_cols = ['intern_id','performance_label','performance_score']
X = df.drop(columns=drop_cols)
yc = df['performance_label']
yr = df['performance_score']

X_train, X_test, yc_train, yc_test = train_test_split(X, yc, test_size=0.2, random_state=42, stratify=yc)
_, _, yr_train, yr_test             = train_test_split(X, yr, test_size=0.2, random_state=42)

# Train best models
print("\n  Training Gradient Boosting Classifier...")
clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.85, random_state=42)
clf.fit(X_train, yc_train)
print(f"  Classifier Accuracy: {clf.score(X_test, yc_test):.4f}")

print("  Training Gradient Boosting Regressor...")
reg = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.85, random_state=42)
reg.fit(X_train, yr_train)

from sklearn.metrics import r2_score
print(f"  Regressor R²:        {r2_score(yr_test, reg.predict(X_test)):.4f}")

# Save everything
pickle.dump(clf,       open('model/model_clf.pkl','wb'))
pickle.dump(reg,       open('model/model_reg.pkl','wb'))
pickle.dump(le_gender, open('model/le_gender.pkl','wb'))
pickle.dump(le_dept,   open('model/le_dept.pkl','wb'))
pickle.dump(le_tier,   open('model/le_tier.pkl','wb'))
pickle.dump(list(X.columns), open('model/feature_cols.pkl','wb'))

print("\n  ✅ Models saved:")
print("     model/model_clf.pkl")
print("     model/model_reg.pkl")
print("     model/le_gender.pkl")
print("     model/le_dept.pkl")
print("     model/le_tier.pkl")
print("     model/feature_cols.pkl")
print("\n  Now run: python app.py")
print("=" * 55)
