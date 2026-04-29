from flask import Flask, request, jsonify
import joblib
import json
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('wildfire_model.pkl')
with open('feature_cols.json') as f:
    feature_cols = json.load(f)

def preprocess(raw_input):
    # same preprocess function from score.py
    discovery_acres = raw_input['discovery_acres']
    fire_cause = raw_input['fire_cause']
    fire_cause_g = raw_input['fire_cause_general']
    fire_behavi = raw_input['fire_behavior']
    fire_beha2 = raw_input['fire_behavior_2']
    fire_beha3 = raw_input['fire_behavior_3']
    primary_fue = raw_input['primary_fuel']
    secondary_f = raw_input['secondary_fuel']
    poo_state = raw_input['state']
    gacc = raw_input['gacc']
    poo_landown = raw_input['land_owner']
    discovery_date = raw_input['discovery_date']

    month = pd.to_datetime(discovery_date).month
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    season = season_map[month]

    row = pd.Series(0, index=feature_cols, dtype=float)
    row['DiscoveryA'] = np.log1p(discovery_acres)
    row['FireBehavi'] = fire_behavi

    for col_name, value in [
        ('POOState', poo_state),
        ('GACC', gacc),
        ('POOLandown', poo_landown),
        ('FireCause', fire_cause),
        ('FireCauseG', fire_cause_g),
        ('season', season)
    ]:
        col = f'{col_name}_{value}'
        if col in row.index:
            row[col] = 1

    for beh in [fire_beha2, fire_beha3]:
        fb_col = f'FB_{beh}'
        if fb_col in row.index:
            row[fb_col] += 1

    for fuel in [primary_fue, secondary_f]:
        fc_col = f'FC_{fuel}'
        if fc_col in row.index:
            row[fc_col] += 1

    return row.values.reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_row = preprocess(data)
        prediction = model.predict(input_row)[0]
        probabilities = model.predict_proba(input_row)[0]
        class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        descriptions = {
            'A': '0.25 acres or less',
            'B': '0.25 to 10 acres',
            'C': '10 to 100 acres',
            'D': '100 to 300 acres',
            'E': '300 to 1,000 acres',
            'F': '1,000 to 5,000 acres',
            'G': '5,000+ acres'
        }
        predicted_label = class_labels[int(prediction)]
        return jsonify({
            "predicted_class": predicted_label,
            "description": descriptions[predicted_label],
            "class_probabilities": dict(zip(class_labels, probabilities.round(3).tolist()))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)