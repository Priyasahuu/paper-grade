## Overview

This project is an AI-based decision support system developed for the Honeywell Hackathon. It predicts process deviations during paper grade transitions, recommends corrective actions before quality variables go out of specification, and discovers hidden relationships between process variables using historical process data.

The solution combines machine learning, correlation analysis, explainable AI, and an interactive dashboard to assist operators in making better decisions during grade changes.

---

## Problem Statement

During paper grade transitions, multiple process variables change simultaneously, increasing the risk of producing off-spec paper and extending stabilization time.

The objective of this project is to:

- Predict future Basis Weight deviations
- Predict future off-spec conditions
- Recommend corrective process setpoints
- Discover hidden process correlations
- Explain model predictions
- Collect operator feedback for continuous improvement

---

## Solution Architecture

```
Synthetic Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
 ┌───────────────┬───────────────┐
 ▼                               ▼
Future Off-Spec Model      Future BW Model
(XGBoost Classifier)      (XGBoost Regressor)
        │
        ▼
Recommendation Engine
        │
        ▼
Correlation Discovery
        │
        ▼
Explainability (SHAP)
        │
        ▼
Streamlit Dashboard
        │
        ▼
Operator Feedback
```

---

## Project Structure

```
paper-grade-intelligence/

app.py
requirements.txt
README.md

analytics/
data/
models/
pages/
results/
saved_models/
```

---

## Dataset

A synthetic industrial dataset was generated to simulate paper machine grade transitions.

Process variables include:

- Stock Flow
- Filler Flow
- Steam Pressure
- Machine Speed
- Headbox Pressure
- Basis Weight
- Moisture
- Ash
- Caliper
- Target Basis Weight
- Future Basis Weight
- Future OffSpec
- Alarm
- Operator Action
- Recommendation Source
- Stabilization Time

---

## Machine Learning Models

### Future Off-Spec Prediction

Model:
XGBoost Classifier

Target:
Future_OffSpec

Purpose:
Predict whether Basis Weight will exceed specification limits before stabilization.

---

### Future Basis Weight Prediction

Model:
XGBoost Regressor

Target:
Future_Basis_Weight

Purpose:
Predict the future Basis Weight trajectory during grade transition.

---

## Analytics

The analytics module identifies relationships among process variables and generates:

- Correlation Matrix
- Strong Correlations
- Hidden Relationships
- Feature Importance
- SHAP Feature Importance
- SHAP Summary Plot

---

## Recommendation Engine

The recommendation engine generates corrective actions such as:

- Increase Stock Flow
- Reduce Stock Flow
- Increase Steam Pressure
- Reduce Machine Speed
- Reduce Filler Flow

Each recommendation includes:

- Predicted Future Basis Weight
- Predicted Off-Spec Status
- Recommendation Source
- Expected Outcome

---

## Dashboard Modules

### Dashboard

Displays live process trends and quality variables.

### Predictions

Displays predicted Basis Weight and future off-spec risk.

### Correlation Analysis

Visualizes hidden relationships and correlation heatmaps.

### Recommendations

Displays AI-generated recommendations and allows operator acceptance or rejection.

### Feedback

Stores operator decisions and comments for future evaluation.

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Output Files

The project generates:

- Correlation Matrix
- Feature Importance
- Hidden Relationship Analysis
- Recommendation Results
- SHAP Analysis
- Operator Feedback
- Prediction Results

---

## Technologies Used

- Python
- Streamlit
- XGBoost
- SHAP
- Plotly
- Pandas
- NumPy
- Scikit-learn

---

## Future Enhancements

- Integration with Honeywell QCS
- Real-time sensor data streaming
- Reinforcement Learning for adaptive control
- Digital Twin integration
- Predictive maintenance module

---

## Developed For

Honeywell Hackathon

Paper Grade Change Intelligence for Paper Manufacturing
