================================================================== Statistical Modeling ========================================================================


### -------------------------------------------To balance the Non outbreak catagory = SMOTNC -----------------------------------------------###

# Python code
# Dataset

df = pd.read_csv("C:\\Users\\HPC\\Downloads\\HPAI_Climate_data.csv")

Index(['Event.ID', 'Disease', 'Serotype', 'Region', 'Subregion', 'Country',
       'States', 'Region.1', 'Locality', 'Latitude', 'Longitude',
       'Observation_date', 'Report_date', 'Year', 'Month', 'Day',
       ' UVA_Irradiance (kW-hr/m^2/day) ', 'UVB_Irradiance (kW-hr/m^2/day) ',
       'UV_Index (W m-2 x 40) ', 'Wind_Speed at 2 Meters (m/s) ',
       'Temperature at 2 Meters (C) ', 'Temperature at 2 Meters Maximum (C) ',
       'Temperature at 2 Meters Minimum (C) ',
       'Specific Humidity at 2 Meters (g/kg) ',
       'Relative Humidity at 2 Meters (%) ',
       'Precipitation Corrected (mm/day) ', 'Surface Pressure (kPa) ',
       'Wind Speed at 10 Meters (m/s) ',
       'Wind Direction at 10 Meters (Degrees) ', 'Diagnosis.status',
       'Animal_type', 'Species', 'Season', 'Status', 'Y']

Status
Outbreak        460
Non_Outbreak     44



import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.impute import SimpleImputer
from collections import Counter

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("C:\\Users\\HPC\\Downloads\\HPAI_Climate_data.csv")
df.columns = df.columns.str.strip()  # remove spaces

# Rename columns
df.rename(columns={
    "UVA_Irradiance (kW-hr/m^2/day)": "UVA_Irradiance",
    "UVB_Irradiance (kW-hr/m^2/day)": "UVB_Irradiance",
    "UV_Index (W m-2 x 40)": "UV_Index",
    "Wind_Speed at 2 Meters (m/s)": "Wind_2M",
    "Temperature at 2 Meters (C)": "Temp",
    "Temperature at 2 Meters Maximum (C)": "Temp_Max",
    "Temperature at 2 Meters Minimum (C)": "Temp_Min",
    "Specific Humidity at 2 Meters (g/kg)": "Spec_Humidity",
    "Relative Humidity at 2 Meters (%)": "Rel_Humidity",
    "Precipitation Corrected (mm/day)": "Precipitation",
    "Surface Pressure (kPa)": "Pressure",
    "Wind Speed at 10 Meters (m/s)": "Wind_10M",
    "Wind Direction at 10 Meters (Degrees)": "Wind_Dir_10M"
}, inplace=True)

target = "Status"

numeric_features = [
    "UVA_Irradiance", "UVB_Irradiance", "UV_Index",
    "Wind_2M", "Temp", "Temp_Max", "Temp_Min",
    "Spec_Humidity", "Rel_Humidity", "Precipitation",
    "Pressure", "Wind_10M", "Wind_Dir_10M", "Year"
]

categorical_features = ["Month", "States"]  # predictors only

X = df[numeric_features + categorical_features].copy()
y = df[target].copy()

# -----------------------------
# 2. Impute numeric features
# -----------------------------
num_imputer = SimpleImputer(strategy='median')
X[numeric_features] = num_imputer.fit_transform(X[numeric_features])

# -----------------------------
# 3. Encode categorical predictors
# -----------------------------
le_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# -----------------------------
# 4. Identify categorical indices for SMOTENC
# -----------------------------
categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

# -----------------------------
# 5. SMOTENC oversampling
# -----------------------------
counter = Counter(y)
n_majority = counter['Outbreak']  
target_minority = int(n_majority * 0.7)

sampling_strategy = {'Non_Outbreak': target_minority}  

smote_nc = SMOTENC(
    categorical_features=categorical_indices,
    sampling_strategy=sampling_strategy,
    random_state=42,
    k_neighbors=5
)

X_res, y_res = smote_nc.fit_resample(X, y)
print(Counter(y_res))

# Decode categorical variables back
for col, le in le_dict.items():
    X_res[col] = le.inverse_transform(X_res[col].round().astype(int))

# Combine features + target
df_balanced = X_res.copy()
df_balanced[target] = y_res

df_balanced.head()

# Save to CSV
df_balanced.to_csv("df_balanced_non_outbreaks3.csv", index=False)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Note - use balanced data (df_balanced_non_outbreaks3.csv) for statistical modeling

# ===============================================================================================
# Multicolinearity check
# ===============================================================================================

# R code

library(dplyr)
library(caret)
library(car)
library(corrplot)
library(pROC)

#  Load data and select variables
df <- read.csv("C:\\Users\\HPC\\Documents\\H5N1\\df_balanced_non_outbreaks3.csv",
               stringsAsFactors = FALSE)


# Subset dataframe
df_selected <- df[, selected_vars]

#  Correlation matrix for numeric predictors
numeric_vars <- df_selected[, sapply(df_selected, is.numeric)]
cor_matrix <- cor(numeric_vars, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)

# Convert categorical variable 'Season' to factor
df_selected$Season <- as.factor(df_selected$Season)

#  Fit a logistic regression model for VIF calculation
glm_model <- glm(Status ~ ., data = df_selected, family = binomial)

#  Calculate VIF
vif_values <- vif(glm_model)
print(vif_values)

#  Updated variable list
selected_vars_reduced <- c(
  "UV_Index", "Wind_2M", "Temp", "Rel_Humidity",
  "Precipitation", "Pressure", "Wind_10M", "Wind_Dir_10M", "Season", "Status"
)

#  Subset the dataframe
df_reduced <- df[, selected_vars_reduced]

#  convert categorical variable is a factor
df_reduced$Season <- as.factor(df_reduced$Season)

#  Fit logistic regression model
glm_model_reduced <- glm(Status ~ ., data = df_reduced, family = binomial)

#  Compute VIF
vif_values_reduced <- vif(glm_model_reduced)
print(vif_values_reduced)



#============================================================================================================================================================================#
# --------------------------------------------------------------------------- GLM pipline ------------------------------------------------------------------------------------#
#=============================================================================================================================================================================#

# Note - performed statistical model on selected variables using multicollinearity check

library(caret)
library(glmnet)
library(pROC)

# Load and prepare data
df <- read.csv("C:\\Users\\HPC\\Documents\\H5N1\\df_balanced_non_outbreaks3.csv",
               stringsAsFactors = FALSE)
head(df)

selected_vars_reduced <- c(
  "UV_Index", "Wind_2M", "Temp", "Rel_Humidity",
  "Precipitation", "Pressure", "Wind_10M", "Wind_Dir_10M",
  "Season", "Status"
)

df_reduced <- df[, selected_vars_reduced]

df_reduced$Season <- as.factor(df_reduced$Season)
head(df_reduced)

set.seed(123)
train_index <- createDataPartition(df_reduced$Status, p = 0.7, list = FALSE)
train_df <- df_reduced[train_index, ]
test_df  <- df_reduced[-train_index, ]

train_df$Status <- factor(train_df$Status, levels = c(0,1), labels = c("No","Yes"))
test_df$Status  <- factor(test_df$Status,  levels = c(0,1), labels = c("No","Yes"))


# Convert to numeric matrix for glmnet
x_train <- model.matrix(Status ~ ., train_df)[, -1]
y_train <- ifelse(train_df$Status == "Yes", 1, 0)

x_test  <- model.matrix(Status ~ ., test_df)[, -1]
y_test  <- ifelse(test_df$Status == "Yes", 1, 0)


# ------  Baseline GLM (no penalty) ------#

glm_base <- glm(Status ~ ., data = train_df, family = binomial)
summary(glm_base)
AIC(glm_base)


# ------ GLM Lasso (alpha = 1) -------#

cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")
glm_lasso <- glmnet(x_train, y_train, alpha = 1, lambda = cv_lasso$lambda.min)
coef(glm_lasso)
coef(cv_lasso, s = "lambda.min")
plot(cv_lasso)


# -------- GLM Ridge (alpha = 0)----------#

cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, family = "binomial")
glm_ridge <- glmnet(x_train, y_train, alpha = 0, lambda = cv_ridge$lambda.min)


#  --------- GLM Elastic Net (alpha = 0.5) ----------#

cv_enet <- cv.glmnet(x_train, y_train, alpha = 0.5, family = "binomial")
glm_enet <- glmnet(x_train, y_train, alpha = 0.5, lambda = cv_enet$lambda.min)


# Evaluation metrics 

compute_glm_metrics <- function(model, type, x_test, test_df, model_name, threshold=0.5) {
  
  # Predictions
  if (type == "glm") {
    probs <- predict(model, test_df, type="response")
    model_aic <- AIC(model)
  } else {
    probs <- predict(model, x_test, type="response")
    probs <- as.numeric(probs)   
    model_aic <- NA
  }
  
  preds <- ifelse(probs >= threshold, "Yes", "No")
  preds <- factor(preds, levels=c("No","Yes"))
  
  cm <- confusionMatrix(preds, test_df$Status, positive="Yes")
  auc <- roc(test_df$Status, probs)$auc
  
  list(
    Model = model_name,
    Accuracy = cm$overall["Accuracy"],
    Precision = cm$byClass["Precision"],
    Recall = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    AUC = auc,
    AIC = model_aic,
    ConfusionMatrix = cm$table
  )
}

m1 <- compute_glm_metrics(glm_base, "glm", x_test, test_df, "GLM Baseline")
m2 <- compute_glm_metrics(glm_lasso, "glmnet", x_test, test_df, "GLM Lasso")
m3 <- compute_glm_metrics(glm_ridge, "glmnet", x_test, test_df, "GLM Ridge")
m4 <- compute_glm_metrics(glm_enet, "glmnet", x_test, test_df, "GLM ElasticNet")

results_glm <- data.frame(
  Model = c(m1$Model, m2$Model, m3$Model, m4$Model),
  Accuracy = c(m1$Accuracy, m2$Accuracy, m3$Accuracy, m4$Accuracy),
  Precision = c(m1$Precision, m2$Precision, m3$Precision, m4$Precision),
  Recall = c(m1$Recall, m2$Recall, m3$Recall, m4$Recall),
  Specificity = c(m1$Specificity, m2$Specificity, m3$Specificity, m4$Specificity),
  AUC = c(m1$AUC, m2$AUC, m3$AUC, m4$AUC),
  AIC = c(m1$AIC, m2$AIC, m3$AIC, m4$AIC)
)

print(results_glm)


#--------------  All 4 Model ROC Curves --------------#

# predicted prob base GLM
prob_glm <- predict(glm_base, newdata = test_df, type = "response")

# predicted prob LASSO
prob_lasso <- predict(
  cv_lasso,
  newx = x_test,
  s = "lambda.min",
  type = "response"
)
prob_lasso <- as.numeric(prob_lasso)

# predicted prob Ridge
prob_ridge <- predict(
  cv_ridge,
  newx = x_test,
  s = "lambda.min",
  type = "response"
)
prob_ridge <- as.numeric(prob_ridge)

# predicted prob Enet
prob_elastic <- predict(
  cv_enet,
  newx = x_test,
  s = "lambda.min",
  type = "response"
)
prob_elastic <- as.numeric(prob_elastic)


stopifnot(
  length(prob_glm) == length(y_test),
  length(prob_lasso) == length(y_test),
  length(prob_ridge) == length(y_test),
  length(prob_elastic) == length(y_test)
)

# All in one plot ROC Curve
library(pROC)

roc_glm     <- roc(y_test, prob_glm,     levels = c(0,1), direction = "<")
roc_lasso   <- roc(y_test, prob_lasso,   levels = c(0,1), direction = "<")
roc_ridge   <- roc(y_test, prob_ridge,   levels = c(0,1), direction = "<")
roc_elastic <- roc(y_test, prob_elastic, levels = c(0,1), direction = "<")

plot(roc_glm, col = "black", lwd = 2, legacy.axes = TRUE,
     main = "ROC Curves for GLM and Penalized Models")
plot(roc_lasso,   col = "blue",  lwd = 2, add = TRUE)
plot(roc_ridge,   col = "red",   lwd = 2, add = TRUE)
plot(roc_elastic, col = "green", lwd = 2, add = TRUE)

abline(a = 0, b = 1, lty = 2, col = "gray")

legend("bottomright",
       legend = c(
         paste0("GLM (AUC = ", round(auc(roc_glm), 3), ")"),
         paste0("LASSO (AUC = ", round(auc(roc_lasso), 3), ")"),
         paste0("Ridge (AUC = ", round(auc(roc_ridge), 3), ")"),
         paste0("ElasticNet (AUC = ", round(auc(roc_elastic), 3), ")")
       ),
       col = c("black", "blue", "red", "green"),
       lwd = 2)


#------------- ROC - Best model -------------#

plot(roc_lasso,
     col = "blue",
     lwd = 2,
     legacy.axes = TRUE,
     main = paste0("ROC Curve – GLM LASSO (AUC = ",
                   round(auc(roc_lasso), 3), ")"))
abline(a = 0, b = 1, lty = 2, col = "gray")

legend("bottomright",
       legend = c(
         paste0("LASSO (AUC = ", round(auc(roc_lasso), 3), ")")
       ),
       col = c( "red"),
       lwd = 2)


--------------------------------------------------------------------------------------------------------------------------------------------------------------

### Non linearity check for GAM 

# Residuals vs fitted values (GLOBAL check)

res_glm <- residuals(glm_base, type = "deviance")
fit_glm <- fitted(glm_base)

plot(fit_glm, res_glm,
     xlab = "Fitted values",
     ylab = "Deviance residuals",
     main = "Residuals vs Fitted (Baseline GLM)")
abline(h = 0, col = "red", lwd = 2)


# Predictor vs residual plots - for each continous predictor
par(mfrow = c(2, 4))
for (v in c("UV_Index", "Wind_2M", "Temp", "Precipitation", "Rel_Humidity",
            "Pressure", "Wind_10M", "Wind_Dir_10M")) {
  
  plot(train_df[[v]], res_glm,
       xlab = v,
       ylab = "Deviance residuals",
       main = paste("Residuals vs", v))
  
  lines(lowess(train_df[[v]], res_glm),
        col = "blue", lwd = 2)
}
par(mfrow = c(1,1))


# partial residuals

library(car)
crPlots(glm_base)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#==========================================================================================================================================================================================#
# ------------------------------------------------------------------------------- GAM pipline ---------------------------------------------------------------------------------------------#
#==========================================================================================================================================================================================#


# 1) GAM (Baseline)


library(mgcv)

gam_base <- gam(
  Status ~
    s(UV_Index, k = 10) +
    s(Temp, k = 10) +
    s(Rel_Humidity, k = 7) +
    s(Pressure, k = 10) +
    s(Wind_10M, k = 7) +
    Wind_2M +
    Precipitation +
    Season,
  data = train_df,
  family = binomial,
  method = "REML",
  gamma = 1.4
)

summary(gam_base)

gam.check(gam_base)



# 2) Lasso-like GAM (variable selection)

gam_lasso <- gam(
  Status ~ 
    s(UV_Index, k = 18) +
    s(Temp, k = 12) +
    s(Rel_Humidity, k = 8) +
    s(Pressure, k = 18) +
    s(Wind_10M, k = 8) +
    Wind_2M +
    Precipitation +
    Season,
  data = train_df,
  family = binomial,
  method = "REML",
  select = TRUE,     # L1 penalty
  gamma = 1.4
)
summary(gam_lasso)

gam.check(gam_lasso)




# 3) Ridge-like GAM (extra smoothing)

gam_ridge <- gam(
  Status ~ 
    s(UV_Index, k = 12) +
    s(Temp, k = 12) +
    s(Rel_Humidity, k = 12) +
    s(Pressure, k = 12) +
    s(Wind_10M, k = 12) +
    Wind_2M +
    Precipitation +
    Season, 
  data = train_df,
  family = binomial,
  method = "REML",
  gamma = 2.0        # strong smoothing 
)
summary(gam_ridge)
gam.check(gam_ridge)



# 4) Elastic Net-like GAM

gam_enet <- gam(
  Status ~ 
    s(UV_Index, k = 15) +
    s(Temp, k = 15) +
    s(Rel_Humidity, k = 10) +
    s(Pressure, k = 15) +
    s(Wind_10M, k = 10) +
    Wind_2M +
    Precipitation +
    Season, 
  data = train_df,
  family = binomial,
  method = "REML",
  select = TRUE,      # L1
  gamma = 1.8         # L2-like smoothing
)
summary(gam_enet)
gam.check(gam_enet)


# Evaluation Metrics 

library(pROC)
library(caret)

compute_metrics <- function(model, test_df, model_name, threshold = 0.5) {
  
  # Predicted probabilities
  preds_prob <- predict(model, newdata = test_df, type = "response")
  
  # Convert to Yes/No based on threshold
  preds_class <- ifelse(preds_prob >= threshold, "Yes", "No")
  preds_class <- factor(preds_class, levels = c("No","Yes"))
  
  # Confusion matrix
  cm <- confusionMatrix(preds_class, test_df$Status, positive = "Yes")
  
  # AUC
  auc <- roc(test_df$Status, preds_prob)$auc
  
  # Output
  list(
    Model = model_name,
    Accuracy = cm$overall["Accuracy"],
    Precision = cm$byClass["Precision"],
    Recall = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    AUC = auc,
    ConfusionMatrix = cm$table
  )
}


m1 <- compute_metrics(gam_base,  test_df, "GAM Baseline")
m2 <- compute_metrics(gam_lasso, test_df, "GAM Lasso")
m3 <- compute_metrics(gam_ridge, test_df, "GAM Ridge")
m4 <- compute_metrics(gam_enet,  test_df, "GAM ElasticNet")


results_table <- data.frame(
  Model = c(m1$Model, m2$Model, m3$Model, m4$Model),
  Accuracy = c(m1$Accuracy, m2$Accuracy, m3$Accuracy, m4$Accuracy),
  Precision = c(m1$Precision, m2$Precision, m3$Precision, m4$Precision),
  Recall = c(m1$Recall, m2$Recall, m3$Recall, m4$Recall),
  Specificity = c(m1$Specificity, m2$Specificity, m3$Specificity, m4$Specificity),
  AUC = c(m1$AUC, m2$AUC, m3$AUC, m4$AUC)
)

print(results_table)


m1$ConfusionMatrix
m2$ConfusionMatrix
m3$ConfusionMatrix
m4$ConfusionMatrix

# ----------- ROC Plot - all in one ------------#

library(pROC)

# Predicted probabilities
preds_prob_list <- list(
  "GAM Baseline"   = predict(gam_base,  test_df, type = "response"),
  "GAM Lasso"      = predict(gam_lasso, test_df, type = "response"),
  "GAM Ridge"      = predict(gam_ridge, test_df, type = "response"),
  "GAM ElasticNet" = predict(gam_enet,  test_df, type = "response")
)

# Actual response
actual <- test_df$Status

# Compute ROC curves
roc_list <- lapply(preds_prob_list, function(p) roc(actual, p))

# Extract AUC values
auc_values <- sapply(roc_list, function(r) round(auc(r), 3))

# Base ROC plot 
plot(roc_list[[1]], col = "red", lwd = 2, main = "ROC Curves for All GAM Models")

# Add remaining ROC curves
plot(roc_list[[2]], col = "blue", lwd = 2, add = TRUE)
plot(roc_list[[3]], col = "green", lwd = 2, add = TRUE)
plot(roc_list[[4]], col = "purple", lwd = 2, add = TRUE)

# Create legend with AUC values
legend_labels <- paste0(names(roc_list), " (AUC=", auc_values, ")")
legend("bottomright",
       legend = legend_labels,
       col = c("red","blue","green","purple"),
       lwd = 2,
       cex = 0.8)   # legend text size


# ---------------- best model --------------#

library(pROC)

# Compute ROC for GAM Baseline
roc_base <- roc(test_df$Status, predict(gam_base, test_df, type = "response"))
auc_base <- round(auc(roc_base), 3)

# Plot ROC curve
plot(roc_base, col = "red", lwd = 2,
     main = "ROC Curve for GAM Baseline",
     legacy.axes = TRUE)

# Add legend with AUC and set font size
legend("bottomright",
       legend = paste0("GAM Baseline (AUC=", auc_base, ")"),
       col = "red",
       lwd = 2,
       cex = 1)   # adjust font size

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


### Best Model - GAM - Baseline

#=================================================================================================================================================================================#
#------------------------------------------------------- Feature importance plot for best model - base GAM ------------------------------------------------------------------------#
#==================================================================================================================================================================================#

gam_base <- gam(
  Status ~
    s(UV_Index, k = 10) +
    s(Temp, k = 10) +
    s(Rel_Humidity, k = 7) +
    s(Pressure, k = 10) +
    s(Wind_10M, k = 7) +
    Wind_2M +
    Precipitation +
    Season,
  data = train_df,
  family = binomial,
  method = "REML",
  gamma = 1.4
)

# old variable names -> new custom names
custom_names <- c(
  "UV_Index"       = "UV Index",
  "Temp"           = "Temperature",
  "Rel_Humidity"   = "Relative Humidity",
  "Pressure"       = "Surface Pressure",
  "Wind_10M"       = "Wind Speed 10m",
  "Wind_2M"        = "Wind Speed 2m",
  "Precipitation"  = "Precipitation",
  "SHAP_SeasonMonsoon" = "Season: Monsoon",
  "SHAP_SeasonSummer"  = "Season: Summer",
  "SHAP_SeasonWinter"  = "Season: Winter"
)


# Smooth terms only (exclude linear/Season terms)
shap_smooth <- as.data.frame(predict(gam_base, type = "terms"))
shap_smooth <- shap_smooth[, !grepl("Wind_2M|Precipitation|Season", names(shap_smooth))]
names(shap_smooth) <- gsub("s\\(|\\)", "", names(shap_smooth))  


coef_all <- coef(gam_base)

shap_linear <- data.frame(
  Wind_2M       = train_df$Wind_2M * coef_all["Wind_2M"],
  Precipitation = train_df$Precipitation * coef_all["Precipitation"]
)

# Model matrix for Season
season_mm <- model.matrix(~ Season, data = train_df)

# Coefficients for Season
season_coef_all <- coef_all[grep("Season", names(coef_all))]

# Multiply each column by its coefficient
shap_season <- as.data.frame(season_mm[, names(season_coef_all), drop = FALSE] %*% diag(season_coef_all))

# Rename to meaningful names
names(shap_season) <- paste0("SHAP_", names(season_coef_all))

shap_full <- cbind(shap_smooth, shap_linear, shap_season)

imp <- colMeans(abs(shap_full))

imp_df <- data.frame(
  Variable = names(imp),
  Importance = imp
)

# Order by importance
imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]

# Plot
library(ggplot2)

ggplot(imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue", width = 0.6) +
  coord_flip() +
  labs(
    title = "Feature Importance from GAM",
    x = "Variables",
    y = "Mean Absolute Contribution"
  ) +
  scale_x_discrete(labels = custom_names) +  # Apply custom names
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size = 13, face = "bold"),
    axis.text = element_text(size = 11, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
  )

ggsave(
  filename = "C:/Users/HPC/Downloads/HPAI_Feature_Importance.png",
  width = 8,
  height = 8,
  dpi = 600,
  bg = "white"
)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#=============================================================================================================================================================================#
#------------------------------------------------------------------- Ponit based Risk Mapping using best model----------------------------------------------------------------#
#=============================================================================================================================================================================#

# Data
setindia_grid <- st_read("C:/Users/HPC/Desktop/Virocon/0.4_Inida_grid/India_grid_0.4_shp.shp")
raw_df <- read.csv("C:/Users/HPC/Downloads/HPAI_Climate_data.csv")

#  Load required libraries
library(mgcv)
library(sf)
library(ggplot2)
library(ggspatial)
library(mgcv)

gam_base <- gam(
  Status ~
    s(UV_Index, k = 10) +
    s(Temp, k = 10) +
    s(Rel_Humidity, k = 7) +
    s(Pressure, k = 10) +
    s(Wind_10M, k = 7) +
    Wind_2M +
    Precipitation +
    Season,
  data = train_df,
  family = binomial,
  method = "REML",
  gamma = 1.4
)

summary(gam_base)

gam.check(gam_base)


#  Load India grid shapefile
india_grid <- st_read("C:/Users/HPC/Desktop/Virocon/0.4_Inida_grid/India_grid_0.4_shp.shp")
india_grid$ST_NM <- toupper(india_grid$ST_NM)  # standardize names


#  Load raw climate data
raw_df <- read.csv("C:/Users/HPC/Downloads/HPAI_Climate_data.csv")


#  Fix column names
colnames(raw_df)[names(raw_df) == "UV_Index..W.m.2.x.40."] <- "UV_Index"
colnames(raw_df)[names(raw_df) == "Wind_Speed.at.2.Meters..m.s."] <- "Wind_2M"
colnames(raw_df)[names(raw_df) == "Temperature.at.2.Meters..C."] <- "Temp"
colnames(raw_df)[names(raw_df) == "Relative.Humidity.at.2.Meters...."] <- "Rel_Humidity"
colnames(raw_df)[names(raw_df) == "Precipitation.Corrected..mm.day."] <- "Precipitation"
colnames(raw_df)[names(raw_df) == "Surface.Pressure..kPa."] <- "Pressure"
colnames(raw_df)[names(raw_df) == "Wind.Speed.at.10.Meters..m.s."] <- "Wind_10M"
colnames(raw_df)[names(raw_df) == "Wind.Direction.at.10.Meters..Degrees."] <- "Wind_Dir_10M"


#  Keep only GAM variables + coordinates
raw_df <- raw_df[, c(
  "UV_Index","Temp","Rel_Humidity","Pressure","Wind_10M",
  "Wind_2M","Precipitation","Season","Longitude","Latitude"
)]

#  Handle missing values
num_cols <- sapply(raw_df, is.numeric)
raw_df[num_cols] <- lapply(
  raw_df[num_cols],
  function(x) { x[is.na(x)] <- median(x, na.rm = TRUE); x }
)

# Match Season levels to training data
raw_df$Season <- factor(raw_df$Season, levels = levels(train_df$Season))

#  Predict outbreak probability (GAM)
raw_df$pred_prob <- predict(gam_base, newdata = raw_df, type = "response")
summary(raw_df$pred_prob)

#  Save predicted probability with coordinates

pred_out <- raw_df[, c("Latitude", "Longitude", "pred_prob")]

write.csv(
  pred_out,
  "C:/Users/HPC/Downloads/GAM_predicted_probability_latlong.csv",
  row.names = FALSE
)

# Define quantile thresholds
q <- quantile(raw_df$pred_prob,
              probs = c(0, 0.2, 0.6, 0.9, 1),
              na.rm = TRUE)

raw_df$risk_class <- cut(
  raw_df$pred_prob,
  breaks = q,
  labels = c("Low", "Medium-Low", "Medium-High", "High"),
  include.lowest = TRUE
)

table(raw_df$risk_class)


raw_sf <- st_as_sf(raw_df, coords = c("Longitude", "Latitude"), crs = 4326)

if (st_crs(raw_sf) != st_crs(india_grid)) {
  raw_sf <- st_transform(raw_sf, st_crs(india_grid))
}

# Plot
ggplot() +
  geom_sf(
    data = india_grid,
    fill = "lightblue",
    color = "lightblue",      
    linewidth = 0.01        
  ) +
  geom_sf(
    data = raw_sf,
    aes(fill = risk_class, color = risk_class),
    shape = 21,
    size = 1.5,
    alpha = 0.9
  ) +
  scale_fill_manual(
    values = c(
      "Low"         = "#FFBF00",
      "Medium-Low"  = "darkorange",
      "Medium-High" = "#FF4848",
      "High"        = "darkred"
    ),
    name = "Predicted Risk",
    guide = guide_legend(
      override.aes = list(
        size = 5,        
        alpha = 1
      )
    )
  ) +
  scale_color_manual(
    values = c(
      "Low"         = "#f4bc1c",
      "Medium-Low"  = "darkorange",
      "Medium-High" = "#FF4848",
      "High"        = "darkred"
    ),
    guide = "none"
  ) +
  annotation_north_arrow(
    location = "tr",
    which_north = "true",
    style = north_arrow_fancy_orienteering
  ) +
  coord_sf() +
  theme_minimal(base_size = 16) +
  labs(title = "HPAI Outbreak Risk Map") +
  theme(
    panel.background  = element_blank(),
    plot.background   = element_blank(),
    legend.background = element_blank(),
    legend.key        = element_blank(),
    plot.title        = element_text(hjust = 0.5, face = "bold"),
    legend.position   = "right",
    panel.grid        = element_blank(),
    axis.title        = element_blank(),
    axis.text         = element_blank(),
    axis.ticks        = element_blank()
  )


ggsave(
  filename = "C:/Users/HPC/Downloads/HPAI_Outbreak_Risk_Map.png",
  width = 10,
  height = 12,
  dpi = 600,
  bg = "white"
)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#=============================================================================================================================================================================#
#------------------------------------------------------------------- Grid based High risk probability  map----------------------------------------------------------------#
#=============================================================================================================================================================================#

# Load Libraries
library(sf)
library(dplyr)
library(ggplot2)
library(sf)
library(ggspatial)

# Data
df = read.csv("C:\\Users\\HPC\\Downloads\\GAM_predicted_probability_latlong.csv")
head(df)
summary(df$pred_prob)

# Filter rows with predicted probability > 0.9
high_risk <- df[df$pred_prob > 0.9, ]

# Check first few rows
head(high_risk)

# Save to CSV
write.csv(
  high_risk,
  "C:/Users/HPC/Downloads/high_risk_points.csv",
  row.names = FALSE
)


india_grid <- st_read("C:/Users/HPC/Desktop/Virocon/0.4_Inida_grid/India_grid_0.4_shp.shp")

# Convert points to sf
df_sf <- st_as_sf(df, coords = c("Longitude", "Latitude"), crs = 4326)

# Spatial join: assign points to grid
joined <- st_join(india_grid, df_sf, join = st_contains)

# Aggregate by grid
grid_prob <- joined %>%
  group_by(row_number()) %>%
  summarise(
    mean_prob = mean(pred_prob, na.rm = TRUE)
  )

#Plot
ggplot() +
  # Grid polygons colored by predicted probability
  geom_sf(
    data = grid_prob,
    aes(fill = mean_prob),
    color = NA
  ) +
  
  # Continuous color scale (4-color gradient)
  scale_fill_gradientn(
    colors = c("#FFBF00", "darkorange", "#FF4848", "darkred"),
    values = scales::rescale(c(0, 0.33, 0.66, 1)),
    na.value = "#CAE9F5",
    name = "Probability of Risk"
  ) +
  
  # India boundary overlay
  geom_sf(
    data = india_grid,
    fill = NA,
    color = "#CAE9F5",
    size = 0.0002
  ) +
  
  # North arrow (direction sign)
  annotation_north_arrow(
    location = "tr",
    which_north = "true",
    style = north_arrow_fancy_orienteering
  ) +
  
  coord_sf(xlim = c(68, 98), ylim = c(6, 38), expand = FALSE) +
  
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    panel.grid       = element_blank(),   
    legend.position  = "right",
    plot.title       = element_text(face = "bold")
  )


ggsave(
  filename = "C:/Users/HPC/Downloads/HPAI_Probability_RisK.png",
  width = 12,
  height = 10,
  dpi = 600,
  bg = "white"
)











































































