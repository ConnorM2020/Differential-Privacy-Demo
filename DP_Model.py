# DP_Model.py
# This would rerun your actual Flask app
import DP_Model

import numpy as np
import pandas as pd
import math
import os
import random
from flask import Flask, jsonify, request, render_template, redirect, url_for
from faker import Faker


TOWN_COORDS = {
    # Ireland
    'Belfast': (54.5973, -5.9301),
    'Dublin': (53.3498, -6.2603),
    'Cork': (51.8985, -8.4756),
    'Limerick': (52.6640, -8.6231),
    'Galway': (53.2707, -9.0568),
    'Waterford': (52.2593, -7.1101),
    'Drogheda': (53.7179, -6.3561),
    'Dundalk': (54.0000, -6.4167),
    'Kilkenny': (52.6541, -7.2448),
    'Sligo': (54.2766, -8.4761),

    # UK
    'London': (51.5074, -0.1278),
    'Manchester': (53.4808, -2.2426),
    'Birmingham': (52.4862, -1.8904),
    'Glasgow': (55.8642, -4.2518),
    'Liverpool': (53.4084, -2.9916),
    'Leeds': (53.8008, -1.5491),
    'Sheffield': (53.3811, -1.4701),
    'Newcastle': (54.9784, -1.6174),
    'Cardiff': (51.4816, -3.1791),
    'Bristol': (51.4545, -2.5879)
}


# ---------------------------
# Synthetic Dataset
# ---------------------------

fake = Faker()
# filtered_data = ""
# noisy_data
def generate_synthetic_patient_data(n=100, seed=None):
    fake = Faker()
    if seed is not None:
        np.random.seed(seed)
        Faker.seed(seed)

    patients = []
    for _ in range(n):
        town = np.random.choice(list(TOWN_COORDS.keys()))
        lat, lon = TOWN_COORDS[town]
        patients.append({
            'forename': fake.first_name(),
            'surname': fake.last_name(),
            'age': np.random.randint(20, 70),
            'gender': np.random.choice(['Male', 'Female']),
            'town': town,
            'town_lat': lat,
            'town_lon': lon,
            'diagnosis': np.random.choice(['Diabetes', 'Hypertension', 'Healthy']),
            'address': fake.address().replace('\n', ', '),
            'email': fake.email(),
            'phone_number': fake.phone_number()
        })

    return pd.DataFrame(patients)


data = generate_synthetic_patient_data(n=100, seed=42)  # Optional seed for reproducibility

# ---------------------------
# Laplace Mechanism
# ---------------------------
def laplace_noise(scale: float) -> float:
    u = np.random.uniform(-0.5, 0.5)
    return -scale * np.sign(u) * np.log(1 - 2 * abs(u))

def add_laplace_noise(value: float, epsilon: float, sensitivity: float = 1.0) -> float:
    scale = sensitivity / epsilon
    return value + laplace_noise(scale)

# ---------------------------
# Differential Privacy Engine
# ---------------------------
class DPEngine:
    def __init__(self, epsilon: float):
        self.original_epsilon = epsilon
        self.epsilon = epsilon
        self.remaining_budget = epsilon

    def reset(self):
        self.remaining_budget = self.original_epsilon

    def _use_budget(self, cost: float):
        if cost > self.remaining_budget:
            raise ValueError("Privacy budget exceeded.")
        self.remaining_budget -= cost

    def private_count(self, df: pd.DataFrame, condition: pd.Series, epsilon: float) -> float:
        self._use_budget(epsilon)
        count = condition.sum()
        return add_laplace_noise(count, epsilon)

    def private_average(self, df: pd.DataFrame, column: str, condition: pd.Series, epsilon: float) -> float:
        self._use_budget(epsilon)
        values = df[condition][column]
        avg = values.mean() if not values.empty else 0.0
        sensitivity = (values.max() - values.min()) / len(values) if len(values) > 0 else 1.0
        return add_laplace_noise(avg, epsilon, sensitivity)
    
    def private_min(self, values: pd.Series, epsilon: float) -> float:
        self._use_budget(epsilon)
        return add_laplace_noise(values.min(), epsilon, sensitivity=5)  # sensitivity estimated

    def private_max(self, values: pd.Series, epsilon: float) -> float:
        self._use_budget(epsilon)
        return add_laplace_noise(values.max(), epsilon, sensitivity=5)
    
    def private_std(self, values: pd.Series, epsilon: float) -> float:
        self._use_budget(epsilon)
        std_dev = values.std()
        return add_laplace_noise(std_dev, epsilon, sensitivity=10)
    
# ---------------------------
# Flask Web App
# ---------------------------
app = Flask(__name__, template_folder=os.getcwd())
dp_engine = DPEngine(epsilon=1.0)


@app.route('/regenerate', methods=['GET'])
def regenerate():
    global data
    data = generate_synthetic_patient_data(n=100)
    dp_engine.reset()
    return redirect(url_for('index'))


@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    filtered_data = None
    noisy_data = None

    if request.method == 'POST':
        town = request.form['town']
        diagnosis = request.form['diagnosis']
        epsilon_count = float(request.form['epsilon_count'])
        epsilon_avg = float(request.form['epsilon_avg'])

        filtered_data = data[(data['town'] == town) & (data['diagnosis'] == diagnosis)]

        noisy_data = filtered_data.copy()
        noisy_data['age'] = noisy_data['age'].apply(lambda x: round(add_laplace_noise(x, epsilon_avg, sensitivity=50), 2))

        try:
            count_condition = data['town'] == town
            count_result = dp_engine.private_count(data, count_condition, epsilon_count)

            avg_condition = data['diagnosis'] == diagnosis
            avg_result = dp_engine.private_average(data, 'age', avg_condition, epsilon_avg)

            min_age = dp_engine.private_min(data['age'], epsilon_avg)
            max_age = dp_engine.private_max(data['age'], epsilon_avg)
            std_age = dp_engine.private_std(data['age'], epsilon_avg)

            results = {
                'town': town,
                'diagnosis': diagnosis,
                'epsilon_count': epsilon_count,
                'epsilon_avg': epsilon_avg,
                'count': round(count_result, 2),
                'avg_age': round(avg_result, 2),
                'min_age': round(min_age, 2),
                'max_age': round(max_age, 2),
                'std_age': round(std_age, 2),
                'remaining': round(dp_engine.remaining_budget, 2)
            }

        except ValueError as e:
            results = {
                'count': str(e),
                'avg_age': '-',
                'min_age': '-',
                'max_age': '-',
                'std_age': '-',
                'remaining': round(dp_engine.remaining_budget, 2)
            }
    
    # Backend sending full data to front end HTML
    patient_coords = data.iloc[::-1].head(10)[['forename', 'surname', 'age', 'gender', 'town', 'town_lat', 'town_lon', 'diagnosis', 'email', 'phone_number', 'address']].to_dict(orient='records')

    return render_template('index.html', results=results,
        sample_data=data.iloc[::-1].reset_index(drop=True),
        raw_filtered=filtered_data if filtered_data is not None else None,
        dp_filtered=noisy_data if noisy_data is not None else None, 
        patient_coords=patient_coords
    )

@app.route('/noisy_data')
def noisy_data():
    epsilon = float(request.args.get('epsilon', 0.5))
    noisy_df = data.copy()

    # Add DP noise to age
    noisy_df['age'] = noisy_df['age'].apply(lambda x: round(add_laplace_noise(x, epsilon, sensitivity=50), 2))

    selected_fields = [
        'forename', 'surname', 'age', 'gender', 'town',
        'town_lat', 'town_lon', 'diagnosis', 'address', 'email', 'phone_number'
    ]

    return jsonify({
        'columns': selected_fields,
        'records': noisy_df[selected_fields].head(10).to_dict(orient='records')
    })


# Allow new users to be created and added to the system. 
@app.route('/signup', methods=['POST'])
def signup():
    forename = request.form['forename']
    surname = request.form['surname']
    age = float(request.form['age'])
    gender = request.form['gender']
    town = request.form['town']
    diagnosis = request.form['diagnosis']
    email = request.form['email']
    phone = request.form['phone']
    address = request.form['address']
    epsilon = float(request.form['epsilon_signup'])
    privacy_level = request.form['privacy_level']

    # Noise sensitivity settings based on user privacy level
    privacy_sensitivity = {
        'high': 0.08,
        'medium': 0.04,
        'low': 0.01
    }
    coord_sensitivity = privacy_sensitivity.get(privacy_level, 0.04)

    noisy_age = round(add_laplace_noise(age, epsilon, sensitivity=50), 2)
    base_lat, base_lon = TOWN_COORDS.get(town, (None, None))
    noisy_lat = round(add_laplace_noise(base_lat, epsilon, coord_sensitivity), 5)
    noisy_lon = round(add_laplace_noise(base_lon, epsilon, coord_sensitivity), 5)

    new_row = {
        'forename': forename,
        'surname': surname,
        'age': noisy_age,
        'gender': gender,
        'town': town,
        'town_lat': noisy_lat,
        'town_lon': noisy_lon,
        'diagnosis': diagnosis,
        'address': address,
        'email': email,
        'phone_number': phone
    }

    global data
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
    return redirect(url_for('index'))




@app.route('/reset', methods=['GET'])
def reset():
    dp_engine.reset()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
