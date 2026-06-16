# HPAI Risk Modelling

A reproducible predictive modelling framework that integrates relevant risk factors, including spatio-temporal and environmental variables, to estimate the risk of HPAI H5N1 and H5N8 outbreaks in India using multiple machine learning approaches. 

---

## Overview

HPAI outbreaks pose significant threats to:

- Commercial poultry production
- Backyard poultry systems
- Wildlife conservation
- Food security
- Public health preparedness

The proposed approach provides a robust tool for risk mapping, early warning and geographically targeted intervention strategies in India. 

---

## Objectives

Develop a predictive modelling framework that integrates relevant risk factors, including spatio-temporal and environmental variables, to estimate the risk of HPAI H5N1 and H5N8 outbreaks in India using multiple machine learning approaches. 

---

## Repository Structure

```text
hpai-risk-modelling/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
├── src/
├── tests/
├── docs/
├── outputs/
├── configs/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Data Sources

Potential data inputs include:

### Disease Surveillance

- Poultry outbreak reports
- Laboratory-confirmed HPAI cases
- Mortality reports
- Wildlife surveillance records

### Environmental Data

- Temperature
- Rainfall
- Humidity
- Wind patterns
- Vegetation indices

### Spatial Data

- Wetlands
- Lakes and rivers
- Land cover
- Poultry density maps
- Administrative boundaries

### Wildlife Data

- Migratory bird flyways
- 

---

## Modelling Approaches

### Statistical Models

- Generalized Linear Models (GLM)
- Generalized Additive Models (GAM)
- LASSO, RIDGE and ELASTICNET

### Machine Learning Models

- Random Forest
- XGBoost
- XGBoost + Neural Network Feature Embeddings
- BRT
- GCN
- GAT


### Spatial Models

- Spatial Autocorrelation Analysis
- Kernel Density Estimation
- Geostatistical Risk Mapping
- Spatio-temporal Models

---

## Workflow

```text
Data Collection
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Risk Mapping
      ↓
Scenario Analysis
      ↓
Decision Support Outputs
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/<your-org>/hpai-risk-modelling.git
cd hpai-risk-modelling
```

### Create Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Explore Data

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Train Model

```bash
python src/models/random_forest.py
```

### Generate Risk Maps

```bash
python src/visualization/maps.py
```

---

## Example Risk Factors

The framework can incorporate variables such as:

| Category | Example Variables |
|-----------|------------------|
| Poultry | Farm density, flock size |
| Wildlife | Migratory bird abundance |
| Environment | Temperature, rainfall |
| Geography | Distance to wetlands |
| Biosecurity | Farm management indicators |
| Temporal | Seasonality, migration period |

---

## Outputs

### Model Outputs

- Outbreak probability
- Risk scores
- Feature importance rankings
- Model performance metrics

### Spatial Outputs

- Risk maps
- Hotspot maps
- Surveillance priority zones
- Interactive dashboards

### Reports

- Risk assessments
- Model evaluation summaries
- Scenario analyses

---

## Evaluation Metrics

Models may be evaluated using:

- ROC-AUC
- Precision
- Recall
- F1 Score
- Brier Score
- Calibration Curves
- Spatial Validation Metrics

---

## Reproducibility

To support reproducible research:

- Version-controlled code
- Configuration-driven workflows
- Documented data transformations
- Automated testing
- Continuous integration

---

## Future Development

Planned enhancements include:

- Real-time surveillance integration
- Satellite remote sensing features
- Climate scenario modelling
- Bayesian forecasting
- Early warning systems
- Interactive web dashboard
- Explainable AI (SHAP)
- Multi-country risk assessments

---

## Contributing

Contributions are welcome.

Suggested areas:

- Data ingestion pipelines
- Epidemiological modelling
- Spatial analytics
- Visualization
- Documentation
- Testing

Please open an issue or submit a pull request.

---

## Disclaimer

This repository is intended for research and decision-support purposes. Model outputs should not be used as the sole basis for disease control decisions without expert review and validation.

---

## License

Released under the MIT License.

---

## Citation

If you use this repository in research, please cite:

```text
HPAI Risk Modelling Project.
Version 1.0.
GitHub Repository.
```
