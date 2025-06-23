# DP_Model.py

import numpy as np
import pandas as pd
import math
import random
from flask import Flask, request, render_template_string

# ---------------------------
# Synthetic Dataset
# ---------------------------
data = pd.DataFrame({
    'age': np.random.randint(20, 70, size=100),
    'gender': np.random.choice(['Male', 'Female'], size=100),
    'town': np.random.choice(['Belfast', 'Dublin', 'Cork'], size=100),
    'diagnosis': np.random.choice(['Diabetes', 'Hypertension', 'Healthy'], size=100)
})

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
        self.epsilon = epsilon
        self.remaining_budget = epsilon

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

# ---------------------------
# Flask Web App
# ---------------------------
app = Flask(__name__)
dp_engine = DPEngine(epsilon=1.0)

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Differential Privacy Demo</title>
    <style>
        .info {
            background: #f9f9f9;
            border: 1px solid #ccc;
            padding: 10px;
            margin-top: 20px;
        }
    </style>
    <script src="/static/epsilon_ui.js"></script>
</head>
<body>
    <h2>DP Query Interface</h2>
    <form method="post">
        Town: <input type="text" name="town"><br>
        Diagnosis: <input type="text" name="diagnosis"><br>
        Epsilon for Count: <input type="number" step="0.01" name="epsilon_count"><br>
        Epsilon for Average Age: <input type="number" step="0.01" name="epsilon_avg"><br>
        <input type="submit" value="Submit">
    </form>
    {% if results %}
        <h3>Results</h3>
        <p><b>DP Count:</b> {{ results['count'] }}</p>
        <p><b>DP Avg Age:</b> {{ results['avg_age'] }}</p>
        <p><b>Remaining Budget:</b> {{ results['remaining'] }}</p>
    {% endif %}

    <div class="info">
        <h4>What is Epsilon (ε)?</h4>
        <p>Epsilon controls how much random noise is added to your query. Lower values mean stronger privacy but more noisy results.</p>
        <ul>
            <li><b>ε = 0.1:</b> Very private, very noisy</li>
            <li><b>ε = 1.0:</b> Balanced privacy and accuracy</li>
            <li><b>ε = 3.0:</b> Less private, more accurate</li>
        </ul>
        <p>Try comparing the same query with ε = 0.1 and ε = 1.0 to see the effect.</p>
    </div>

    <div>
        <h4>Valid Inputs:</h4>
        <ul>
            <li><b>Towns:</b> Belfast, Dublin, Cork</li>
            <li><b>Diagnoses:</b> Diabetes, Hypertension, Healthy</li>
            <li><b>Epsilon (ε):</b> Between 0.01 and 1.0 (smaller = more private, more noise)</li>
        </ul>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        town = request.form['town']
        diagnosis = request.form['diagnosis']
        epsilon_count = float(request.form['epsilon_count'])
        epsilon_avg = float(request.form['epsilon_avg'])

        try:
            count_condition = data['town'] == town
            count_result = dp_engine.private_count(data, count_condition, epsilon_count)

            avg_condition = data['diagnosis'] == diagnosis
            avg_result = dp_engine.private_average(data, 'age', avg_condition, epsilon_avg)

            results = {
                'count': round(count_result, 2),
                'avg_age': round(avg_result, 2),
                'remaining': round(dp_engine.remaining_budget, 2)
            }
        except ValueError as e:
            results = {'count': str(e), 'avg_age': '-', 'remaining': dp_engine.remaining_budget}

    return render_template_string(HTML_FORM, results=results)

if __name__ == '__main__':
    app.run(debug=True)
