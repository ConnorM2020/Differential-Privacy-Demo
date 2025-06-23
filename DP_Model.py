# DP_Model.py

import numpy as np
import pandas as pd
import math
import random
from flask import Flask, request, render_template_string, redirect, url_for

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
        .summary-section {
            display: flex;
            gap: 60px;
            justify-content: center;
            align-items: flex-start;
        }
        .summary-section ul {
            margin-top: 0;
        }
    </style> 
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
    <div class="info">
      <h3>Query Summary</h3>
      <div class="summary-section">
        <div>
          <p><b>Inputs:</b></p>
          <ul>
            <li>Town: <code>{{ results['town'] }}</code></li>
            <li>Diagnosis: <code>{{ results['diagnosis'] }}</code></li>
            <li>Epsilon (Count): <code>{{ results['epsilon_count'] }}</code></li>
            <li>Epsilon (Average): <code>{{ results['epsilon_avg'] }}</code></li>
          </ul>
        </div>
        <div>
          <p><b>Outputs:</b></p>
          <ul>
            <li>DP Count: <code>{{ results['count'] }}</code></li>
            <li>DP Avg Age: <code>{{ results['avg_age'] }}</code></li>
            <li>DP Min Age: <code>{{ results['min_age'] }}</code></li>
            <li>DP Max Age: <code>{{ results['max_age'] }}</code></li>
            <li>DP Std Dev: <code>{{ results['std_age'] }}</code></li>
            <li>Remaining Budget: <code>{{ results['remaining'] }}</code></li>
          </ul>
        </div>
      </div>
      {% if results['remaining'] == 0.0 %}
      <form method="get" action="/reset">
          <button type="submit">Reset Privacy Budget</button>
      </form>
      {% endif %}
    </div>
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
            results = {'count': str(e), 'avg_age': '-', 'remaining': round(dp_engine.remaining_budget, 2)}

    return render_template_string(HTML_FORM, results=results)

@app.route('/reset', methods=['GET'])
def reset():
    dp_engine.reset()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
