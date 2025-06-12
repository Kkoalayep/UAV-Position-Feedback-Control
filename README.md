# Group 8 - Reinforcement Learning Based PID Control for Tello UAV

## 📌 Project Overview

This project implements a reinforcement learning (RL) framework to tune a PID controller for a simulated Tello drone using PyBullet. The goal is to improve flight control performance by automatically optimizing PID parameters for both yaw and xyz axes through RL training.

---

## 🗂️ Folder Structure

```
Group 8/
│
├── controller.py              # Main PID controller script
├── run.py                     # Manual run using fixed PID values
├── result.txt                 # Summary of performance
├── targets.csv                # Target positions for training/testing
├── README.md                  # Project documentation (this file)
│
├── Lab5.2/                    # Lab 5.2 experiment data and visualization
│   ├── controller.py          # Controller used in Lab 5.2
│   ├── controller_log_*.txt   # Experimental logs from Lab 5.2
│   ├── Log2Plot.py            # Script to convert logs into plots
│   ├── log_1.png              # Output plot of experimental data
│   ├── First Flight.MOV       # Video for the first flight (successful)
│   └── Last Flight.MOV        # Video for the first flight (failed)
│
├── pid_rl/                    # RL training code
│   ├── train_pid_rl.py        # Training for xyz PID
│   ├── train_pid_yaw.py       # Training for yaw PID
│   ├── run_best.py            # Evaluate best models
│   ├── pid_policy.py          # Policy network
│   └── rl_pid_env.py          # RL environment for PID
│
├── model/                     # Saved models and training curves
│   ├── best_pid_yaw.pth
│   ├── best_pid_xyz.pth
│   ├── pid_yaw_reward_curve.png
│   └── pid_xyz_reward_curve.png
│
├── src/                       # Drone control and PID classes
│   ├── PID_controller.py
│   └── tello_controller.py
│
└── resources/                 # URDF and mesh files
    ├── tello.urdf
    ├── tello_cover.stl
    ├── tello_v2.stl
    └── prop.stl
```

---

## ▶️ How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train PID (Yaw)**:
   ```bash
   python pid_rl/train_pid_yaw.py
   ```

3. **Train PID (XYZ)**:
   ```bash
   python pid_rl/train_pid_rl.py
   ```

4. **Run best models**:
   ```bash
   python pid_rl/run_best.py
   ```

5. **Manual control test**:
   ```bash
   python run.py
   ```

6. **[Lab 5.2] Visualize experimental results**:
   ```bash
   cd Lab5.2
   python Log2Plot.py
   ```

---

## ⚙️ Requirements

Make sure you're using **Python 3.8 or later**.

Example `requirements.txt`:
```
torch
numpy
matplotlib
pybullet
```

---

## 👤 Authors

Group 8 – [Jiahao Lin]

---

## 📌 Notes

- The `model/` directory contains all trained models and learning curves.
- The `Lab5.2/` folder includes experimental data logs and the corresponding plotting utility.
