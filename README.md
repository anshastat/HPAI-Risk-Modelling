# HPAI Risk Modelling

A reproducible framework for modelling and forecasting **Highly Pathogenic Avian Influenza (HPAI)** risk in poultry and wild bird populations using epidemiological, environmental, spatial, and surveillance data.

---

## Overview

HPAI outbreaks pose significant threats to:

- Commercial poultry production
- Backyard poultry systems
- Wildlife conservation
- Food security
- Public health preparedness

This project provides a modular platform for:

- Risk assessment
- Outbreak prediction
- Spatial hotspot identification
- Environmental driver analysis
- Scenario simulation
- Decision support for surveillance and intervention planning

---

## Objectives

### Primary Objectives

- Estimate outbreak risk at farm, district, regional, or national scales
- Identify environmental and ecological risk factors
- Produce spatial risk maps
- Support targeted surveillance programs
- Evaluate intervention strategies

### Research Objectives

- Understand relationships between:
  - Wild bird populations
  - Poultry density
  - Climate variables
  - Water bodies and wetlands
  - Land-use patterns
- Develop explainable predictive models
- Compare statistical and machine learning approaches

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
- Species abundance
- Waterfowl distribution
- Seasonal migration timing

---

## Modelling Approaches

### Statistical Models

- Logistic Regression
- Generalized Linear Models (GLM)
- Bayesian Hierarchical Models

### Machine Learning Models

- Random Forest
- XGBoost
- Gradient Boosting
- LightGBM
- Ensemble Models

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
