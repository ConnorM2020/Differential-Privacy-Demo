**Differential Privacy Dashboard**

A <strong> full-stack interactive dashboard <strong>  demonstrating the application of Differential Privacy (DP) for safeguarding sensitive data while allowing meaningful analysis. 
Built as part of the PETs Summer Internship 2025, this tool visualises privacy-utility trade-offs and simulates re-identification attacks under varying privacy levels.

---
**Overview**

This project showcases how differential privacy mechanisms can be applied to synthetic patient data, allowing users to:

Adjust privacy levels (ε) and observe real-time changes to data distortion.
Visualise location and attribute noise using interactive maps and tables.
Simulate re-identification and membership inference attacks.
Explore the concept of user vulnerability through a colour-coded risk interface.

---

**Key Features**

- **Privacy Sliders**  
  Control attribute, location, and table noise using adjustable ε-values.

- **Interactive Map (Leaflet.js)**  
  Displays noisy patient locations with popups containing obfuscated personal and medical data.

- **Real-Time Accuracy Charts (Chart.js)**  
  Track MAE (Mean Absolute Error) as noise changes for age and geolocation data.

- **Attack Simulation Mode**  
  Simulates adversarial attacks using partial information (age, town, heart rate). Vulnerable users are revealed based on risk score thresholds.

- **Risk Score Visualisation**  
  Users are highlighted in **green (safe)**, **orange (moderate risk)**, or **red (high risk)** depending on their exposure at a given ε-level.

- **Detailed User Modals**  
  View risk details, full user profiles, and simulated tampering alerts.

---

## Technologies Used

- **Backend**: Python (Flask)
- **Frontend**: HTML, CSS, JavaScript
- **Mapping**: Leaflet.js
- **Graphing**: Chart.js
- **Data**: Synthetic patient data generated using `Faker` and `NumPy`

---

Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/differential-privacy-dashboard.git
cd differential-privacy-dashboard
=======
# Differential-Privacy-Demo
