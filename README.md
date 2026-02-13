
# Cric-ML 🏏  

Cric-ML is a cricket analytics and machine learning project built using structured **ball-by-ball data stored in a MySQL database**.

The project is currently in its **initial development stage** and focuses on structured data analysis, visualization, and future prediction use cases.

---

## 🚧 Project Status (Initial Stage)

This project is in the **early phase of development**.

- Currently, **47 matches have been added to the database**
- The dataset is partial and continuously expanding
- The ML prediction module is **under active development**

### 🎯 Current Purpose

At this stage, the primary goals are:

- To design a clean and scalable database schema  
- To validate analytics logic  
- To build reliable player, team, and phase-wise insights  
- To engineer ML-ready features for future prediction models  

All current notebooks and visualizations are based on this structured but limited dataset.

---

## 📊 Current Capabilities

### 🧍 Player-Based Analytics
- Runs, strike rate, dot balls  
- Boundary percentage  
- Phase-wise batting performance  
- Player vs bowling style insights  

### ⏳ Phase-Wise Analysis
- Powerplay performance  
- Middle overs performance  
- Death overs performance  

### 🏏 Team Performance Analysis
- Phase-wise team run rate  
- Structural team strength evaluation  
- Comparative team analysis  

### 🎯 Bowling Analytics
- Economy rate analysis  
- Wickets by phase  
- Pressure indicators  
- Matchup-based effectiveness  

### 📈 Match Flow Visualizations
- Over-by-over performance tracking  
- Phase momentum shifts  
- Tactical turning point exploration  

---

## 🤖 Machine Learning (Under Development)

The machine learning prediction system is currently in the **development and refinement stage**.

Work in progress includes:

- Feature engineering for match prediction  
- Player-phase strength modeling  
- Team structural modeling  
- Matchup-based evaluation  
- Venue and situational feature integration  

⚠️ The prediction engine is still evolving and will improve as more match data is integrated.

---

## 🔮 Future Roadmap

### Phase 1: IPL 2025 Full Data Integration

- Add complete IPL 2025 ball-by-ball data to the database  
- Re-run and validate all analytics on full-season data  
- Improve feature reliability and coverage  
- Strengthen ML training dataset  

---

### Phase 2: T20 World Cup 2026 Prediction

Use IPL 2025 data as a major training source to:

- Build structured prediction models  
- Predict match outcomes  
- Analyze player impact  
- Evaluate team strengths  
- Apply trained models to T20 World Cup 2026 scenarios  

---

### Phase 3: IPL 2026 Prediction

Combine:

- IPL 2025 data  
- T20 World Cup 2026 insights  

To:

- Predict IPL 2026 performance  
- Compare player form across tournaments  
- Improve long-term prediction stability  
- Enhance multi-season modeling capability  

---

## 🛠 Tech Stack

- Python  
- MySQL  
- Pandas  
- Matplotlib  
- Jupyter Notebook  
- XGBoost (ML experimentation)  

---

## 📂 Project Structure

```
notebooks/    → Analysis and visualization notebooks  
src/          → ML pipeline and future production-level code

---

## ⚠️ Note

Since the project is in its early stage and the dataset is partial, all insights and trends are exploratory and evolving. The machine learning models are currently being developed and refined as more data becomes available.

---

## 📌 Goal

To evolve Cric-ML from an analytics-focused project into a robust, structured cricket intelligence and prediction system using real-world T20 datasets.

