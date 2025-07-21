# DP_Model.py
# This would rerun your actual Flask app

import numpy as np
import pandas as pd
import math
import random
import os
from flask import send_from_directory

from flask import Flask, jsonify, request, render_template, redirect, url_for
from faker import Faker

def send_privacy_alert(to_number, message):
    print(f"[SMS DISABLED] Would send to {to_number}: {message}")

# from dotenv import load_dotenv

injected_names = set()  # Global memory tracker

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
            
            # Vital signs
            'heart_rate': np.random.randint(60, 100),
            'systolic_bp': np.random.randint(90, 140),
            'diastolic_bp': np.random.randint(60, 90),
            'temperature': round(np.random.normal(36.6, 0.5), 1),
            'respiratory_rate': np.random.randint(12, 20),
            'oxygen_saturation': np.random.randint(90, 100),
            'bmi': round(np.random.normal(25, 5), 1),
            'blood_glucose': np.random.randint(70, 120),

            # Contact info
            'address': fake.address().replace('\n', ', '),
            'email': fake.email(),
            'phone_number': fake.msisdn()
        })


    return pd.DataFrame(patients)


data = generate_synthetic_patient_data(n=10, seed=42)  # Optional seed for reproducibility

# Assign initial risk scores to first 10 users
def assign_initial_risk_scores(df):
    risk_scores = [round(random.uniform(0.0, 1.0), 2) for _ in range(len(df))]
    df['risk_score'] = risk_scores
    df['status'] = ['At Risk' if score > 0 else 'Protected' for score in risk_scores]

assign_initial_risk_scores(data)


# ---------------------------
# Laplace Mechanism
# ---------------------------
def laplace_noise(scale: float) -> float:
    u = np.random.uniform(-0.5, 0.5)
    return -scale * np.sign(u) * np.log(1 - 2 * abs(u))

def add_laplace_noise(value: float, epsilon: float, sensitivity: float = 1.0) -> float:
    scale = sensitivity / epsilon
    return value + laplace_noise(scale)


def distort_name(name: str, epsilon: float) -> str:
    if not name:
        return name

    # Skip distortion if epsilon is high enough (e.g. ≥ 1.0)
    if epsilon >= 0.99:
        return name

    distortion_level = int((1.0 / max(epsilon, 0.01)) * 1.5)
    special_chars = ['*', '#', '@', '%', '$', '!', '?']

    chars = list(name)
    indices = random.sample(range(len(chars)), min(distortion_level, len(chars)))
    for i in indices:
        if chars[i].isalpha():
            chars[i] = random.choice(special_chars)
    return ''.join(chars)



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
@app.route('/home')
def home():
    return redirect(url_for('index'))


@app.route("/attack_simulation.html")
def serve_attack_simulation():
    return send_from_directory(os.getcwd(), "attack_simulation.html")


@app.route('/regenerate', methods=['GET'])
def regenerate():
    global data
    data = generate_synthetic_patient_data(n=10)
    dp_engine.reset()
    return redirect(url_for('index'))

@app.route('/reset', methods=['GET'])
def reset():
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
    patient_coords = data.iloc[::-1].head(10)[[
        'forename', 'surname', 'age', 'gender', 'town',
        'town_lat', 'town_lon', 'diagnosis',
        'heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature',
        'respiratory_rate', 'oxygen_saturation', 'bmi', 'blood_glucose',
        'address', 'email', 'phone_number'
    ]].to_dict(orient='records')

    return render_template('index.html', results=results,
        sample_data=data.iloc[::-1].reset_index(drop=True),
        raw_filtered=filtered_data if filtered_data is not None else None,
        dp_filtered=noisy_data if noisy_data is not None else None, 
        patient_coords=patient_coords
    )

@app.route('/available_towns', methods=['GET'])
def available_towns():
    towns = sorted(data['town'].unique())
    return jsonify({"towns": towns})


# Add noise to user profile 
@app.route('/noisy_data')
def noisy_data():
    epsilon = float(request.args.get('epsilon', 0.5))
    noisy_df = data.copy()

    if epsilon < 0.99:
        # Apply DP noise to age
        noisy_df['age'] = noisy_df['age'].apply(lambda x: round(add_laplace_noise(x, epsilon, sensitivity=50), 2))

        # Distort names
        noisy_df['forename'] = noisy_df['forename'].apply(lambda x: distort_name(x, epsilon))
        noisy_df['surname'] = noisy_df['surname'].apply(lambda x: distort_name(x, epsilon))
    else:
        # Keep original values when ε is high (1.00)
        pass  # leave age and names as-is

    selected_fields = [
        'forename', 'surname', 'age', 'gender', 'town',
        'town_lat', 'town_lon', 'diagnosis', 'heart_rate', 'systolic_bp', 'diastolic_bp',
        'temperature', 'respiratory_rate', 'oxygen_saturation', 'bmi', 'blood_glucose',
        'address', 'email', 'phone_number'
    ]

    return jsonify({
        'columns': selected_fields,
        'records': noisy_df[selected_fields].head(10).to_dict(orient='records')
    })


# Add noise to location profile 
@app.route('/noisy_location_data')
def noisy_location_data():
    loc_epsilon = float(request.args.get('loc_epsilon', 0.5))
    loc_df = data.copy()

    def add_location_noise(lat, lon):
        if loc_epsilon >= 0.99:
            return lat, lon  # Exact
        lat_noise = laplace_noise(0.05 / loc_epsilon)
        lon_noise = laplace_noise(0.05 / loc_epsilon)
        return round(lat + lat_noise, 5), round(lon + lon_noise, 5)

    loc_df['town_lat'], loc_df['town_lon'] = zip(*loc_df.apply(
        lambda row: add_location_noise(row['town_lat'], row['town_lon']), axis=1))

    selected_fields = [
        'forename', 'surname', 'age', 'gender', 'town',
        'town_lat', 'town_lon', 'diagnosis', 'heart_rate', 'systolic_bp', 'diastolic_bp',
        'temperature', 'respiratory_rate', 'oxygen_saturation', 'bmi', 'blood_glucose',
        'address', 'email', 'phone_number'
    ]

    return jsonify({
        'columns': selected_fields,
        'records': loc_df[selected_fields].head(10).to_dict(orient='records')
    })

def clamp(value, min_val=0, max_val=120):
    return max(min_val, min(value, max_val))

# Injecting a malicious user example Alice for showcase purposes 
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

    noisy_age = clamp(round(add_laplace_noise(age, epsilon, sensitivity=50), 2), 0, 100)
    base_lat, base_lon = TOWN_COORDS.get(town, (None, None))
    noisy_lat = round(add_laplace_noise(base_lat, epsilon, coord_sensitivity), 5)
    noisy_lon = round(add_laplace_noise(base_lon, epsilon, coord_sensitivity), 5)

    try:
        send_privacy_alert(
            to_number=phone,
            message=f"Hi {forename}, you’ve signed up with ε={epsilon:.2f}. Your data is protected using differential privacy!"
        )
    except Exception as e:
        print("Failed to send SMS:", e)

    new_row = {
        'forename': forename,
        'surname': surname,
        'age': noisy_age,
        'gender': gender,
        'town': town,
        'town_lat': noisy_lat,
        'town_lon': noisy_lon,
        'diagnosis': diagnosis,

        # Vital signs
        'heart_rate': np.random.randint(60, 100),
        'systolic_bp': np.random.randint(90, 140),
        'diastolic_bp': np.random.randint(60, 90),
        'temperature': round(np.random.normal(36.6, 0.5), 1),
        'respiratory_rate': np.random.randint(12, 20),
        'oxygen_saturation': np.random.randint(90, 100),
        'bmi': round(np.random.normal(25, 5), 1),
        'blood_glucose': np.random.randint(70, 120),

        # Contact
        'address': address,
        'email': email,
        'phone_number': phone
        
    }

    global data
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

    return redirect(url_for('index'))

# -----------------------------
#   Accuracy Metrics
# ----------------------------
@app.route('/accuracy_metrics')
def accuracy_metrics():
    epsilon = float(request.args.get('epsilon', 0.5))

    # Retrieve or generate both datasets
    original_data = data.to_dict(orient='records')

    # Generate noisy version of age and coordinates
    noisy_df = data.copy()
    noisy_df['age'] = noisy_df['age'].apply(lambda x: round(add_laplace_noise(x, epsilon, sensitivity=50), 2))

    noisy_df['town_lat'], noisy_df['town_lon'] = zip(*noisy_df.apply(
        lambda row: (
            row['town_lat'] if epsilon >= 0.99 else round(row['town_lat'] + laplace_noise(0.05 / epsilon), 5),
            row['town_lon'] if epsilon >= 0.99 else round(row['town_lon'] + laplace_noise(0.05 / epsilon), 5)
        ),
        axis=1
    ))
    noisy_data = noisy_df.to_dict(orient='records')

    if not original_data or not noisy_data or len(original_data) != len(noisy_data):
        return jsonify({"error": "Mismatched or missing data"}), 400

    def mae(val1, val2):
        return sum(abs(a - b) for a, b in zip(val1, val2)) / len(val1)

    try:
        orig_ages = [float(row["age"]) for row in original_data]
        noisy_ages = [float(row["age"]) for row in noisy_data]

        orig_lats = [float(row["town_lat"]) for row in original_data]
        noisy_lats = [float(row["town_lat"]) for row in noisy_data]

        orig_lons = [float(row["town_lon"]) for row in original_data]
        noisy_lons = [float(row["town_lon"]) for row in noisy_data]

        return jsonify({
            "age_mae": mae(orig_ages, noisy_ages),
            "lat_mae": mae(orig_lats, noisy_lats),
            "lon_mae": mae(orig_lons, noisy_lons)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/compute_accuracy', methods=['POST'])
def compute_accuracy():
    data = request.json
    original = data.get("original_data", [])
    noisy = data.get("noisy_data", [])

    if not original or not noisy or len(original) != len(noisy):
        return jsonify({"error": "Mismatched or missing data"}), 400

    def mae(original_vals, noisy_vals):
        return sum(abs(o - n) for o, n in zip(original_vals, noisy_vals)) / len(original_vals)

    # Extract age and coordinates from both
    original_ages = [float(row["Age"]) for row in original]
    noisy_ages = [float(row["Age"]) for row in noisy]

    original_lats = [float(row["Latitude"]) for row in original]
    noisy_lats = [float(row["Latitude"]) for row in noisy]

    original_lons = [float(row["Longitude"]) for row in original]
    noisy_lons = [float(row["Longitude"]) for row in noisy]

    # Compute errors
    age_error = mae(original_ages, noisy_ages)
    lat_error = mae(original_lats, noisy_lats)
    lon_error = mae(original_lons, noisy_lons)

    return jsonify({
        "age_mae": age_error,
        "lat_mae": lat_error,
        "lon_mae": lon_error
    })


# --------------------------
# Simulate Attack
# ------------------------
# Fix this injecting client and check for valid attack
@app.route('/inject_target_user')
def inject_target_user():
    global data, injected_names

    target_name = "Alice Target"
    target_email = "alice.target@example.com"

    # Construct the full target user object
    target_user = {
        'forename': 'Alice',
        'surname': 'Target',
        'age': 64,
        'gender': 'Female',
        'town': 'Galway',
        'town_lat': TOWN_COORDS['Galway'][0],
        'town_lon': TOWN_COORDS['Galway'][1],
        'diagnosis': 'Healthy',
        'heart_rate': 86,
        'systolic_bp': 120,
        'diastolic_bp': 80,
        'temperature': 36.6,
        'respiratory_rate': 16,
        'oxygen_saturation': 98,
        'bmi': 24.5,
        'blood_glucose': 90,
        'address': '123 Target St, Galway',
        'email': target_email,
        'phone_number': '353861234567',
        'status': 'At Risk',
        'risk_score': 0.91
    }

    # If already injected, just return the full user object again
    if target_name in injected_names:
        return jsonify(target_user), 200

    # Check if exists in current dataset
    exists = not data[
        (data['forename'] == 'Alice') & 
        (data['surname'] == 'Target') & 
        (data['email'] == target_email)
    ].empty

    if exists:
        injected_names.add(target_name)
        return jsonify(target_user), 200  # Return full user

    # Inject user once
    injected_names.add(target_name)
    data = pd.concat([data, pd.DataFrame([target_user])], ignore_index=True)

    return jsonify(target_user), 200  # Always return full user object


@app.route('/simulate_attack', methods=['POST'])
def simulate_attack():
    try:
        data_in = request.get_json(force=True)
        attack_type = data_in.get("attack_type")

        epsilon = float(data_in.get("epsilon", 0.5))

        known_age = float(data_in.get("age", -1))
        known_town = data_in.get("town", "")
        known_hr = float(data_in.get("heart_rate", -1))
        print(f"[ATTACK] Type={attack_type} | Age={known_age} | Town={known_town} | HR={known_hr} | ε={epsilon}")

        visible_data = data.tail(101)


        exact_match = visible_data[
            (visible_data['age'] == known_age) &
            (visible_data['town'] == known_town) &
            (visible_data['heart_rate'] == known_hr)
        ]

        if exact_match.empty:
            return jsonify({
                "result": "❌ Invalid attack: Only top 10 visible records are allowed for attack simulation.",
                "vulnerable_users": [],
                "safe_users": [],
                "matched_records": [],
                "match_confidence": 0.0,
                "match_count": 0,
                "dataset_size": len(data),
                "epsilon": epsilon
            }), 400

        # Add DP noise
        simulated_data = data.copy()
        if epsilon < 0.99:
            simulated_data['age'] = simulated_data['age'].apply(lambda x: round(add_laplace_noise(x, epsilon, 50), 2))
            simulated_data['heart_rate'] = simulated_data['heart_rate'].apply(lambda x: round(add_laplace_noise(x, epsilon, 10), 2))

        match_df = simulated_data[
            (simulated_data['town'] == known_town) &
            (simulated_data['age'] == known_age) &
            (simulated_data['heart_rate'] == known_hr)
        ]

        vulnerable_list = []
        safe_list = []
        matched_records = []

        for _, row in data.iterrows():
            is_exact = (
                row['town'] == known_town and
                row['age'] == known_age and
                row['heart_rate'] == known_hr
            )
            is_noisy_match = any(
                (match['email'] == row['email']) for _, match in match_df.iterrows()
            )

            record = {
                "name": f"{row['forename']} {row['surname']}",
                "town": row['town'],
                "age": row['age'],
                "heart_rate": row['heart_rate'],
                "epsilon": epsilon
            }

            if is_exact and is_noisy_match:
                score = max(0.01, min(1.0, 1.0 - epsilon))
                vulnerable_list.append({
                    **record,
                    "status": "At Risk",
                    "risk_score": round(score, 2),
                    "matched": True
                })
                matched_records.append(row.to_dict())
            else:
                safe_list.append({
                    **record,
                    "status": "Protected",
                    "risk_score": 0.0,
                    "matched": False
                })

        match_count = len(matched_records)
        dataset_size = len(data)
        match_confidence = round(match_count / dataset_size, 4) if dataset_size else 0.0

        if attack_type == "membership":
            inferred = "Likely Member" if match_confidence > 0.05 else "Not a Member"
            result_msg = (
                f"{match_count} match(es) found. Membership Inference: <strong>{inferred}</strong> "
                f"(confidence: {match_confidence * 100:.2f}%)"
            )
        else:
            result_msg = (
                f"🧠 {len(vulnerable_list)} user(s) vulnerable to re-identification. "
                f"{len(safe_list)} protected by differential privacy."
            )

        return jsonify({
            "result": result_msg,
            "vulnerable_users": vulnerable_list,
            "safe_users": safe_list,
            "matched_records": matched_records[:5],
            "match_confidence": match_confidence,
            "match_count": match_count,
            "dataset_size": dataset_size,
            "epsilon": epsilon
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/vulnerable_users", methods=["GET"])
def get_vulnerable_users_by_epsilon():
    epsilon = float(request.args.get("epsilon", 0.5))

    # Use first 10 users only (no reversal)
    visible_data = data.head(10).to_dict(orient='records')

    vulnerable_users = []

    for user in visible_data:
        risk_score = user.get("risk_score", 1.0)

        if risk_score >= (1.0 - epsilon):
            vulnerable_users.append({
                "name": f"{user.get('forename', '?')} {user.get('surname', '?')}",
                "town": user.get("town", "?"),
                "age": user.get("age", "?"),
                "heart_rate": user.get("heart_rate", "?"),
                "status": user.get("status", "At Risk"),
                "risk_score": round(risk_score, 2)
            })
    return jsonify(vulnerable_users)

@app.route('/attacked_user')
def attacked_user():
    name = request.args.get('name', 'Unknown')
    town = request.args.get('town', 'Unknown')
    age = request.args.get('age', 'N/A')
    heart_rate = request.args.get('heart_rate', 'N/A')

    # validation:
    try:
        age = int(float(age))
        heart_rate = int(float(heart_rate))
    except ValueError:
        age = "Invalid"
        heart_rate = "Invalid"

    return render_template("attacked_user.html", name=name, town=town, age=age, heart_rate=heart_rate)

if __name__ == '__main__':
    app.run(debug=True)