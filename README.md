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

### Disease Surveillance Data

HPAI outbreak records were obtained from:

FAO EMPRES-i+ Database: https://empres-i.apps.fao.org/
WOAH World Animal Health Information System (WAHIS): https://wahis.woah.org/#/event-management

These databases provide information on outbreak locations, dates, affected species, and disease occurrence.

### Climate and Meteorological Data

Bioclimatic variables were downloaded from:

WorldClim Version 2.1: https://www.worldclim.org/data/worldclim21.html

Meteorological variables were obtained from:

NASA POWER Data Access Viewer: https://power.larc.nasa.gov/data-access-viewer/

### Environmental and Land Cover Data

Environmental variables were obtained from:

NASA Global Land Data Assimilation System (GLDAS): https://ldas.gsfc.nasa.gov/gldas

These datasets provide land surface and environmental characteristics relevant to disease ecology.

### Spatial Data

Administrative boundaries and geographic reference data were obtained from:

OpenStreetMap (OSM): https://www.openstreetmap.org/

Water bodies, rivers, lakes, and other hydrographic features were extracted from OpenStreetMap data downloads available through:

Geofabrik India Extracts: https://download.geofabrik.de/asia/india.html

### Wildlife Data

Migratory bird flyways/Path : 

A migratory bird exposure variable was manually derived based on the proximity of outbreak locations to migratory bird flyways, bird sanctuaries, and wetlands identified from publicly available sources.



## Modelling Approaches

### Statistical Models

- Generalized Linear Models (GLM)
- Generalized Additive Models (GAM)
- LASSO, RIDGE and ELASTIC NET

### Machine Learning Models

- Random Forest
- XGBoost
- XGBoost + Neural Network Feature Embeddings
- BRT
- GCN
- GAT


### Spatial Models

- Spatial Autocorrelation Analysis
- Geostatistical Risk Mapping
- LISA
- ICC (GLMM)

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
