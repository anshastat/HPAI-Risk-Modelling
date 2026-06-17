# HPAI H5N1 and H5N8 Outbreak Risk Modelling

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
: https://www.researchgate.net/figure/Flyway-for-migratory-birds-India-source-Google_fig1_328687081



## Modelling Approaches

### Spatial Models and Analyses

* Global Moran's I Spatial Autocorrelation Analysis
* Local Indicators of Spatial Association (LISA)
* Intraclass Correlation Coefficient (ICC) using Generalized Linear Mixed Models (GLMM)
* Geostatistical Risk Mapping

### Statistical Models

```text
* Generalized Linear Models (GLM)
  
├── Baseline GLM
├── GLM + LASSO
├── GLM + Ridge
└── GLM + Elastic Net

* Generalized Additive Models (GAM)
├── Baseline GAM
├── GAM + LASSO
├── GAM + Ridge
└── GAM + Elastic Net
```


### Machine Learning Models

* Random Forest (RF)
* Extreme Gradient Boosting (XGBoost)
* XGBoost with Neural Network Feature Embeddings
* Boosted Regression Trees (BRT)
* Graph Convolutional Networks (GCN)
* Graph Attention Networks (GAT)




## Workflow

```text
Data Collection
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Exploratory Data Analysis
      ↓
Spatial Analysis
      ↓
Statistical Modelling
      ↓
Machine Learning Modelling
      ↓
Model Evaluation
      ↓
Risk Mapping
```


## Risk Factors

The framework incorporated variables such as:

| Category                  | Variables                                                           |
| ------------------------- | --------------------------------------------------------------------------- |
| **Poultry Population**    | Chicken Density, Duck Density                                               |
| **Wildlife Factors**      | Migratory Bird Flyways                                                      |
| **Climatic Factors**      | Temperature, Precipitation, Wind Speed, Surface Pressure, Relative Humidity |
| **Environmental Factors** | Soil Moisture, LAI, LST, EVI, NDVI, Waterbodies,Soil Moisture, Leaf Area Index (LAI), Land Surface Temperature (LST), Enhanced Vegetation Index (EVI), Normalized Difference Vegetation Index (NDVI), Water Bodies                                         |
| **Spatial Factors**       | Latitude, Longitude, State, District                                        |
| **Temporal Factors**      | Season, Migration Period                                                    |



---

## Outputs

### Model Outputs

- Outbreak Risk Probability
- Feature importance rankings
- Model performance metrics

### Spatial Outputs

- Risk maps
- Hotspot maps
- Surveillance priority zones

### Reports

- Model evaluation summaries
- Discussion

---

## Evaluation Metrics

Models may be evaluated using:

- ROC-AUC
- Precision
- Recall
- F1 Score
- Accuracy
---


## Future Development

Planned enhancements include:

- Real-time surveillance integration
- Early warning systems
- Interactive web dashboard
---

