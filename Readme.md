# 🚦 Intelligent Traffic Signal Optimization Using Reinforcement Learning in SUMO

**Institution:** Indian Institute of Information Technology, Nagpur  
**Mentor:** Dr. Rashmi Pandhare  
**Contributors:** Saurabh Kumar • Darshan Tate • Sai Chandra Mouli  

---

## 📌 Overview
Traffic congestion at urban intersections is worsened by static signal timers that fail to adapt to real-time vehicle flow. This project implements and compares:
- **Static Timer Controller**
- **Tabular Q-Learning RL Agent**
- **Deep Q-Network (DQN) RL Agent**

Using **SUMO simulation**, **TraCI real-time control**, and **E1 lane detectors**, the models are evaluated using standard **ITS metrics**.

---

## 🧠 Reinforcement Learning Design

| Component | Implementation |
|---|---|
| **State Space** | 7 values → 6 lane detector counts + current signal phase |
| **Actions** | `0 = Keep phase`, `1 = Switch to next phase` |
| **Reward** | `R = −(Sum of lane queues)` |

---

## 📊 Model-Wise Statistical Analysis

---

### 🔹 Static Model — Queue Length Output
![Static Queue Length Graph](<images/Staticgraph.jpeg>)

- Avg Queue ≈ **7.77 vehicles**
- Max Queue = **12**
- Total Delay ≈ **23,000 vehicle-seconds**
- Throughput = **1929 vehicles**
- Max Travel Time = **1332 sec**

---

### 🔹 Q-Learning Model — Queue Length Output
![Q-Learning Queue Length Graph](<images/QGraph.jpeg>)

- Avg Queue ↓ **~38%**
- Max Queue = **8**
- Delay ↓ **~39%**
- Throughput = **2368 vehicles** (**+22% over static**)
- Travel Time ↓ **~57%**

---

### 🔹 DQN Model — Queue Length Output
![DQN Queue Length Graph](<images/DQNgraph.jpeg>)

- Avg Queue ↓ **~35%**
- Max Queue = **8**
- Delay ↓ **~36%**
- Throughput = **2303 vehicles**
- Travel Time ↓ **~54%**
- Most stable switching due to neural generalization

---

## ⏱ Time Analysis Outputs (Per Model)

### 🟡 Static Controller — Python Time Analysis
![Static Time Analysis Output](<images/Staticmetrics.jpeg>)

- Minimal compute cost
- No adaptation → high real-world time wasted in queues

---

### 🟢 Q-Learning Controller — Python Time Analysis
![Q-Learning Time Analysis Output](<images/Qmetrics.jpeg>)

- Fast Q-table lookup & updates
- Aggressive phase optimization
- Best overall system-level time savings

---

### 🔵 DQN Controller — Python Time Analysis
![DQN Time Analysis Output](<images/DQN metrics.jpeg>)

- Includes neural inference overhead
- Still better than static due to smarter phase control
- Best worst-case congestion handling

---

## Flow Chart 
![Flow Chart](<images/NoteGPT-Flowchart-1757804082141.png>)

---

## 🏁 Conclusion
Reinforcement Learning enables **real-time adaptive signal control** with clear improvements:

![Conclusion Summary Image](<images/WhatsApp Image 2025-11-19 at 10.11.35 PM.jpg>)

- **Delay Reduction:** **35–40%**
- **Queue Reduction:** **≈38%**
- **Throughput Gain:** **+22% vehicles cleared**
- **Q-Learning:** Best **average efficiency**
- **DQN:** Best **stability & worst-case travel time**
- Edge deployment feasible on **Jetson Nano / Raspberry Pi**

---

## 🔮 Future Scope
- Multi-intersection network
- Multi-Agent RL coordination
- CCTV + YOLO detector replacement
- Emergency & pedestrian priority
- TensorRT edge optimization

---

## 📜 License
MIT License
