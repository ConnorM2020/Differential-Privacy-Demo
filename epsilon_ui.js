// epsilon_ui.js

// This script dynamically explains epsilon selection with visual feedback.
document.addEventListener("DOMContentLoaded", function () {
  const epsilonInput = document.querySelector("input[name='epsilon_count']");
  const epsilonInfo = document.createElement("div");

  epsilonInfo.id = "epsilon-feedback";
  epsilonInfo.style.marginTop = "10px";
  epsilonInput.parentNode.insertBefore(epsilonInfo, epsilonInput.nextSibling);

  epsilonInput.addEventListener("input", function () {
    const val = parseFloat(epsilonInput.value);
    let message = "";
    if (val < 0.2) {
      message = "Very private, but high noise";
    } else if (val < 1.0) {
      message = "Balanced privacy and accuracy";
    } else if (val >= 1.0) {
      message = "More accurate, but weaker privacy";
    } else {
      message = "";
    }
    epsilonInfo.textContent = `ε = ${val.toFixed(2)} → ${message}`;
  });

  const resultCard = document.createElement("div");
  resultCard.className = "info";
  resultCard.style.border = "2px solid #444";
  resultCard.style.marginTop = "20px";
  resultCard.style.padding = "15px";
  resultCard.style.background = "#ffffff";
  resultCard.style.boxShadow = "0 0 10px rgba(0,0,0,0.1)";
  resultCard.innerHTML = `
    <h4>Latest Query Summary</h4>
    <p><b>Input:</b></p>
    <ul>
      <li>Town: <code>Dublin</code></li>
      <li>Diagnosis: <code>Diabetes</code></li>
      <li>Epsilon (Count): <code>0.4</code></li>
      <li>Epsilon (Average): <code>0.6</code></li>
    </ul>
    <p><b>Output:</b></p>
    <ul>
      <li>DP Count: <code>47.07</code></li>
      <li>DP Avg Age: <code>55.52</code></li>
      <p><b>DP Min Age:</b> {{ results['min_age'] }}</p>
      <p><b>DP Max Age:</b> {{ results['max_age'] }}</p>
      <p><b>DP Std Dev:</b> {{ results['std_age'] }}</p>
      <li>Remaining Budget: <code>0.0</code></li>
    </ul>
    <button id="resetBtnTop">🔁 Reset Privacy Budget</button>
  `;
  document.body.appendChild(resultCard);

  const exampleOutput = document.createElement("div");
  exampleOutput.className = "info";
  exampleOutput.innerHTML = `
    <h4>Example Output:</h4>
    <p><b>Input:</b> Town = "Dublin", Diagnosis = "Diabetes", ε_count = 0.4, ε_avg = 0.6</p>
    <p><b>DP Count:</b> 9.78</p>
    <p><b>DP Avg Age:</b> 45.21</p>
    <p><b>Remaining Budget:</b> 0.00</p>
    <canvas id="noiseChart" width="400" height="200"></canvas>
    <button id="resetBtn">🔁 Reset Privacy Budget</button>
  `;
  document.body.appendChild(exampleOutput);

  // ChartJS example to illustrate noise
  const ctx = document.getElementById('noiseChart');
  if (ctx) {
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['True Count', 'DP Count'],
        datasets: [{
          label: 'Count Comparison',
          data: [10, 9.78],
          backgroundColor: ['#4caf50', '#2196f3']
        }]
      },
      options: {
        scales: {
          y: {
            beginAtZero: true,
            max: 60
          }
        }
      }
    });
  }

  // Reset buttons to reload the page (reset state)
  const resetBtn = document.getElementById('resetBtn');
  const resetBtnTop = document.getElementById('resetBtnTop');
  const handleReset = () => window.location.reload();

  if (resetBtn) resetBtn.addEventListener('click', handleReset);
  if (resetBtnTop) resetBtnTop.addEventListener('click', handleReset);
});
