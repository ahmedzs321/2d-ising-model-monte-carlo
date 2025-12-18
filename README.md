# Monte Carlo Simulation of the 2D Ising Model

This repository contains a computational physics project investigating the two-dimensional Ising model using the Metropolis Monte Carlo algorithm. The project demonstrates how macroscopic phase transitions emerge from simple microscopic spin interactions in statistical mechanics.

---

## 📌 Project Overview

The Ising model is a foundational model in statistical physics that describes ferromagnetism using binary spin variables with nearest-neighbor interactions. Although simple in form, the model exhibits rich behavior, including spontaneous magnetization and a continuous phase transition at a critical temperature.

In this project, a square lattice of spins is simulated across a range of temperatures to study:
- Energy per spin
- Magnetization
- Heat capacity
- Magnetic susceptibility
- Spin-domain formation and disorder

Monte Carlo sampling is used to approximate thermodynamic ensemble averages, allowing the system’s phase transition to be visualized and quantified.

---

## 🧠 Methods

- **Model:** 2D Ising model on an L × L square lattice  
- **Algorithm:** Metropolis Monte Carlo  
- **Boundary Conditions:** Periodic  
- **Units:** J = 1, k₍ᴮ₎ = 1, h = 0  
- **Observables:**  
  - Energy  
  - Magnetization  
  - Heat capacity  
  - Susceptibility  

Each temperature point includes equilibration sweeps followed by measurement sweeps to ensure statistically meaningful results.

---

## 📊 Results

The simulation reproduces the expected ferromagnetic–paramagnetic phase transition:

- At **low temperatures**, spins align into an ordered, magnetized phase.
- Near the **critical temperature (T ≈ 2.27)**, large domains and strong
  fluctuations appear.
- At **high temperatures**, the system becomes disordered with zero net
  magnetization.
- Heat capacity and susceptibility exhibit pronounced peaks near the transition.

Sample output plots and spin configuration visualizations are included.

---

## 📁 Repository Structure
```bash
├── 2D_ising_model_monte_carlo.py # Main simulation code
├── Syed_Monte_Carlo_Simulation_of_the_2D_Ising_Model.pdf # Full paper
├── Sample Results/
  ├── Figure 1.png # Spin configurations
  └── Figure 2.png # Thermodynamic observables
├── Sources/ # Reference material
└── README.md
```

---

## ▶ How to Run the Code

Just run the code from inside an IDE or from a folder directly, using powershell and the following command:
```bash
python 2D_ising_model_monte_carlo.py
```
**Requirements:**
- Python 3
- NumPy
- Matplotlib

---

## 📚 References

Key references used in this project include:

- Introductory texts on Monte Carlo simulation
- Lecture notes on the Ising model
- Analytical results by Onsager for the 2D Ising model

Full citations are provided in the accompanying paper.

---

## ✍️ Author

**Ahmed Syed**
Department of Physics
Wayne State University

---

## 📎 Notes

This project was completed as part of coursework in thermodynamics and statistical physics. The Ising model and Monte Carlo methods were chosen to illustrate how numerical simulation complements analytical theory in modern physics.

