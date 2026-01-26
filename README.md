🚗 EV Battery Fault Detection System Using SOC & LSTM

This project detects **faulty or aged Electric Vehicle (EV) batteries**
by analyzing **State of Charge (SOC)** behavior using an **LSTM deep learning model**.

---

📌 Objective
To automatically classify EV batteries as:
- ✅ Healthy (Fresh)
- ⚠️ Faulty (Aged)

based on SOC time-series behavior.

---

🧠 Methodology
1. Load experimental battery data (Fresh & Aged)
2. Estimate SOC using **Coulomb Counting**
3. Apply preprocessing (smoothing / noise reduction)
4. Create time-series sequences
5. Train **LSTM neural network**
6. Classify battery health state

---

🗂 Dataset
- `Experimental_data_fresh_cell.csv`
- `Experimental_data_aged_cell.csv`

Each dataset contains:
- Time
- Voltage
- Current

---

🔬 Technologies Used
- MATLAB R2025b
- LSTM (Deep Learning)
- Signal Processing
- SOC Estimation

---

📊 Results
- Smooth SOC pattern → Healthy battery
- Irregular SOC drop → Faulty battery
- LSTM successfully learns degradation behavior

---

🚀 Applications
- EV Battery Health Monitoring
- Predictive Maintenance
- Battery Management Systems (BMS)

---

👨‍🎓 Developed By
**Suyash Naik **  
Diploma in Computer Science & Information Technology | EV & ML Enthusiast
