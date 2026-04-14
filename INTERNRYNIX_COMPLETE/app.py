"""
app.py - Sprint 4 Flask Web Application
Intern Performance Prediction & Analytics System
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import io
import os

app = Flask(__name__)

# Load models
clf       = pickle.load(open('model/model_clf.pkl', 'rb'))
reg       = pickle.load(open('model/model_reg.pkl', 'rb'))
le_gender = pickle.load(open('model/le_gender.pkl', 'rb'))
le_dept   = pickle.load(open('model/le_dept.pkl',   'rb'))
le_tier   = pickle.load(open('model/le_tier.pkl',   'rb'))
feat_cols = pickle.load(open('model/feature_cols.pkl', 'rb'))

LABEL_MAP    = {0: 'Low', 1: 'Medium', 2: 'High'}
LABEL_COLOR  = {'High': '#16a34a', 'Medium': '#d97706', 'Low': '#dc2626'}
LABEL_ICON   = {'High': '🌟', 'Medium': '📈', 'Low': '⚠️'}

def safe_encode(le, value, default=0):
    try:
        return le.transform([value])[0]
    except:
        return default

def engineer_features(row):
    row['academic_index']    = round((row['cgpa']/10)*0.7 + (row['certifications_count']/8)*0.3, 4)
    row['soft_skill_avg']    = round((row['communication_score']+row['teamwork_rating']+row['initiative_score']+row['adaptability_score']+row['mentor_rating'])/5, 4)
    row['work_efficiency']   = round(row['tasks_completed']/(row['bugs_or_errors']+1), 4)
    row['reliability_score'] = round((row['attendance_pct']/100)*(row['deadline_adherence']/100)*100, 4)
    row['experience_score']  = round(row['internship_exp_months']*0.6 + row['certifications_count']*0.4, 4)
    row['potential_index']   = round(row['academic_index']*30 + row['soft_skill_avg']*25 + row['work_efficiency']*20 + row['reliability_score']*15 + row['experience_score']*10, 4)
    return row

def predict_single(data):
    data['gender']       = safe_encode(le_gender, data['gender'])
    data['department']   = safe_encode(le_dept,   data['department'])
    data['college_tier'] = safe_encode(le_tier,   data['college_tier'])
    data = engineer_features(data)

    X = pd.DataFrame([data])[feat_cols]

    label_idx = clf.predict(X)[0]
    label     = LABEL_MAP[label_idx]
    proba     = clf.predict_proba(X)[0]
    score     = round(float(reg.predict(X)[0]), 2)
    confidence= round(float(proba[label_idx]) * 100, 1)

    return {
        'label'     : label,
        'score'     : score,
        'confidence': confidence,
        'proba'     : {
            'Low'   : round(float(proba[0])*100, 1),
            'Medium': round(float(proba[1])*100, 1),
            'High'  : round(float(proba[2])*100, 1),
        },
        'color' : LABEL_COLOR[label],
        'icon'  : LABEL_ICON[label],
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'age'                   : float(request.form.get('age', 22)),
            'gender'                : request.form.get('gender', 'Male'),
            'department'            : request.form.get('department', 'Engineering'),
            'college_tier'          : request.form.get('college_tier', 'Tier 2'),
            'cgpa'                  : float(request.form.get('cgpa', 7.0)),
            'backlogs'              : float(request.form.get('backlogs', 0)),
            'certifications_count'  : float(request.form.get('certifications_count', 2)),
            'internship_exp_months' : float(request.form.get('internship_exp_months', 0)),
            'attendance_pct'        : float(request.form.get('attendance_pct', 80)),
            'tasks_completed'       : float(request.form.get('tasks_completed', 15)),
            'deadline_adherence'    : float(request.form.get('deadline_adherence', 75)),
            'overtime_hours'        : float(request.form.get('overtime_hours', 5)),
            'bugs_or_errors'        : float(request.form.get('bugs_or_errors', 3)),
            'communication_score'   : float(request.form.get('communication_score', 7)),
            'teamwork_rating'       : float(request.form.get('teamwork_rating', 7)),
            'initiative_score'      : float(request.form.get('initiative_score', 7)),
            'adaptability_score'    : float(request.form.get('adaptability_score', 7)),
            'mentor_rating'         : float(request.form.get('mentor_rating', 7)),
        }
        result = predict_single(data)
        name   = request.form.get('intern_name', 'Intern')
        return render_template('result.html', result=result, name=name, data=data)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files.get('csv_file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
        results = []

        for _, row in df.iterrows():
            data = {
                'age'                   : float(row.get('age', 22)),
                'gender'                : str(row.get('gender', 'Male')),
                'department'            : str(row.get('department', 'Engineering')),
                'college_tier'          : str(row.get('college_tier', 'Tier 2')),
                'cgpa'                  : float(row.get('cgpa', 7.0)),
                'backlogs'              : float(row.get('backlogs', 0)),
                'certifications_count'  : float(row.get('certifications_count', 2)),
                'internship_exp_months' : float(row.get('internship_exp_months', 0)),
                'attendance_pct'        : float(row.get('attendance_pct', 80)),
                'tasks_completed'       : float(row.get('tasks_completed', 15)),
                'deadline_adherence'    : float(row.get('deadline_adherence', 75)),
                'overtime_hours'        : float(row.get('overtime_hours', 5)),
                'bugs_or_errors'        : float(row.get('bugs_or_errors', 3)),
                'communication_score'   : float(row.get('communication_score', 7)),
                'teamwork_rating'       : float(row.get('teamwork_rating', 7)),
                'initiative_score'      : float(row.get('initiative_score', 7)),
                'adaptability_score'    : float(row.get('adaptability_score', 7)),
                'mentor_rating'         : float(row.get('mentor_rating', 7)),
            }
            res = predict_single(data)
            results.append({
                'name'       : str(row.get('intern_name', row.get('intern_id', f'Intern {_+1}'))),
                'label'      : res['label'],
                'score'      : res['score'],
                'confidence' : res['confidence'],
                'color'      : res['color'],
                'icon'       : res['icon'],
            })

        return render_template('batch_result.html', results=results)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for developers"""
    try:
        data = request.get_json()
        result = predict_single(data)
        return jsonify({
            'status'            : 'success',
            'performance_label' : result['label'],
            'performance_score' : result['score'],
            'confidence_pct'    : result['confidence'],
            'probabilities'     : result['proba'],
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
