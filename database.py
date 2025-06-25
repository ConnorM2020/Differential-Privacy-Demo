from flask import Flask, request, jsonify
import random
import numpy as np

app = Flask(__name__)

# Generate 1,000 random patient records
towns = ['Belfast', 'Dublin', 'Cork']
diagnoses = ['Diabetes', 'Hypertension', 'Healthy']

patients = [
    {
        "town": random.choice(towns),
        "diagnosis": random.choice(diagnoses),
        "age": random.randint(18, 90)
    }
    for _ in range(1000)
]

def add_laplace_noise(value, epsilon, sensitivity=1):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

@app.route('/query', methods=['POST'])
def dp_query():
    data = request.json
    town = data.get('town')
    diagnosis = data.get('diagnosis')
    eps_count = float(data.get('eps_count', 1.0))
    eps_avg = float(data.get('eps_avg', 1.0))

    # Filter matching records
    filtered = [p for p in patients if p['town'] == town and p['diagnosis'] == diagnosis]

    true_count = len(filtered)
    true_avg = sum(p['age'] for p in filtered) / true_count if true_count > 0 else 0

    # Apply differential privacy
    noisy_count = max(0, round(add_laplace_noise(true_count, eps_count)))
    noisy_avg = round(add_laplace_noise(true_avg, eps_avg, sensitivity=72), 2)  # age range: ~18–90

    return jsonify({
        "noisy_count": noisy_count,
        "noisy_average_age": noisy_avg
    })

if __name__ == '__main__':
    app.run(debug=True)
