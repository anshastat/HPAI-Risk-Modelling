
#==============================================================================================================
#--------------------------------------- Statistical Model ----------------------------------------------------
#==============================================================================================================



# =============================================================================
# Load Libraries
# =============================================================================

library(dplyr)
library(caret)
library(glmnet)
library(car)
library(corrplot)
library(pROC)

# =============================================================================
# Define Project Paths
# =============================================================================

data_file <- "data/balanced_data.csv"

output_dir <- "outputs/statistical_models"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# =============================================================================
# Load Data
# =============================================================================

df <- read.csv(data_file, stringsAsFactors = FALSE)

str(df)

# =============================================================================
# Variable Selection
# =============================================================================

selected_vars <- c(
  "UV_Index",
  "Wind_2M",
  "Temp",
  "Rel_Humidity",
  "Precipitation",
  "Pressure",
  "Wind_10M",
  "Wind_Dir_10M",
  "Season",
  "Status"
)

df_selected <- df[, selected_vars]


# =============================================================================
# 1. Multicollinearity Assessment
# =============================================================================

# --------------------------------------------------
# 1.1 Correlation Matrix
# --------------------------------------------------

numeric_vars <- df_selected[, sapply(df_selected, is.numeric)]

cor_matrix <- cor(
  numeric_vars,
  use = "complete.obs"
)

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  tl.cex = 0.8
)

# --------------------------------------------------
# 1.2 Variance Inflation Factor (VIF)
# --------------------------------------------------

df_selected$Season <- as.factor(df_selected$Season)

glm_vif <- glm(
  Status ~ .,
  data = df_selected,
  family = binomial
)

vif_values <- vif(glm_vif)

print(vif_values)


# =============================================================================
# Final Reduced Dataset
# =============================================================================

selected_vars_reduced <- c(
  "UV_Index",
  "Wind_2M",
  "Temp",
  "Rel_Humidity",
  "Precipitation",
  "Pressure",
  "Wind_10M",
  "Wind_Dir_10M",
  "Season",
  "Status"
)

df_reduced <- df[, selected_vars_reduced]

df_reduced$Season <- as.factor(df_reduced$Season)

# =============================================================================
# Train-Test Split
# =============================================================================

set.seed(123)

train_index <- createDataPartition(
  df_reduced$Status,
  p = 0.70,
  list = FALSE
)

train_df <- df_reduced[train_index, ]
test_df  <- df_reduced[-train_index, ]

# Convert outcome

train_df$Status <- factor(
  train_df$Status,
  levels = c(0,1),
  labels = c("No","Yes")
)

test_df$Status <- factor(
  test_df$Status,
  levels = c(0,1),
  labels = c("No","Yes")
)

# =============================================================================
# Create Model Matrices
# =============================================================================

x_train <- model.matrix(
  Status ~ .,
  train_df
)[,-1]

y_train <- ifelse(
  train_df$Status == "Yes",
  1,
  0
)

x_test <- model.matrix(
  Status ~ .,
  test_df
)[,-1]

y_test <- ifelse(
  test_df$Status == "Yes",
  1,
  0
)

# =============================================================================
# 2. Generalized Linear Models (GLM)
# =============================================================================

# =============================================================================
# 2.1 Baseline GLM
# =============================================================================

glm_base <- glm(
  Status ~ .,
  data = train_df,
  family = binomial
)

summary(glm_base)

AIC(glm_base)

# =============================================================================
# 2.2 GLM-LASSO
# =============================================================================

cv_lasso <- cv.glmnet(
  x_train,
  y_train,
  alpha = 1,
  family = "binomial"
)

glm_lasso <- glmnet(
  x_train,
  y_train,
  alpha = 1,
  lambda = cv_lasso$lambda.min
)

coef(cv_lasso, s = "lambda.min")

plot(cv_lasso)

# =============================================================================
# 2.3 GLM-Ridge
# =============================================================================

cv_ridge <- cv.glmnet(
  x_train,
  y_train,
  alpha = 0,
  family = "binomial"
)

glm_ridge <- glmnet(
  x_train,
  y_train,
  alpha = 0,
  lambda = cv_ridge$lambda.min
)

# =============================================================================
# 2.4 GLM-Elastic Net
# =============================================================================

cv_enet <- cv.glmnet(
  x_train,
  y_train,
  alpha = 0.5,
  family = "binomial"
)

glm_enet <- glmnet(
  x_train,
  y_train,
  alpha = 0.5,
  lambda = cv_enet$lambda.min
)

# =============================================================================
# Model Evaluation Function
# =============================================================================

compute_glm_metrics <- function(
    model,
    model_type,
    x_test,
    test_df,
    model_name,
    threshold = 0.5
) {
  
  if (model_type == "glm") {
    
    probs <- predict(
      model,
      test_df,
      type = "response"
    )
    
    model_aic <- AIC(model)
    
  } else {
    
    probs <- predict(
      model,
      x_test,
      type = "response"
    )
    
    probs <- as.numeric(probs)
    
    model_aic <- NA
  }
  
  preds <- ifelse(
    probs >= threshold,
    "Yes",
    "No"
  )
  
  preds <- factor(
    preds,
    levels = c("No","Yes")
  )
  
  cm <- confusionMatrix(
    preds,
    test_df$Status,
    positive = "Yes"
  )
  
  auc_value <- roc(
    test_df$Status,
    probs
  )$auc
  
  return(
    data.frame(
      Model = model_name,
      Accuracy = cm$overall["Accuracy"],
      Precision = cm$byClass["Precision"],
      Recall = cm$byClass["Sensitivity"],
      Specificity = cm$byClass["Specificity"],
      AUC = auc_value,
      AIC = model_aic
    )
  )
}

# =============================================================================
# Evaluate Models
# =============================================================================

m1 <- compute_glm_metrics(
  glm_base,
  "glm",
  x_test,
  test_df,
  "GLM Baseline"
)

m2 <- compute_glm_metrics(
  glm_lasso,
  "glmnet",
  x_test,
  test_df,
  "GLM LASSO"
)

m3 <- compute_glm_metrics(
  glm_ridge,
  "glmnet",
  x_test,
  test_df,
  "GLM Ridge"
)

m4 <- compute_glm_metrics(
  glm_enet,
  "glmnet",
  x_test,
  test_df,
  "GLM Elastic Net"
)

results_glm <- rbind(
  m1,
  m2,
  m3,
  m4
)

print(results_glm)

write.csv(
  results_glm,
  file.path(
    output_dir,
    "GLM_Model_Performance.csv"
  ),
  row.names = FALSE
)

# =============================================================================
# ROC Curve Comparison
# =============================================================================

prob_glm <- predict(
  glm_base,
  newdata = test_df,
  type = "response"
)

prob_lasso <- as.numeric(
  predict(
    cv_lasso,
    newx = x_test,
    s = "lambda.min",
    type = "response"
  )
)

prob_ridge <- as.numeric(
  predict(
    cv_ridge,
    newx = x_test,
    s = "lambda.min",
    type = "response"
  )
)

prob_elastic <- as.numeric(
  predict(
    cv_enet,
    newx = x_test,
    s = "lambda.min",
    type = "response"
  )
)

roc_glm <- roc(y_test, prob_glm)
roc_lasso <- roc(y_test, prob_lasso)
roc_ridge <- roc(y_test, prob_ridge)
roc_elastic <- roc(y_test, prob_elastic)

png(
  file.path(
    output_dir,
    "ROC_All_GLM_Models.png"
  ),
  width = 1200,
  height = 900
)

plot(
  roc_glm,
  col = "black",
  lwd = 2,
  main = "ROC Curves"
)

plot(roc_lasso, add = TRUE, col = "blue", lwd = 2)
plot(roc_ridge, add = TRUE, col = "red", lwd = 2)
plot(roc_elastic, add = TRUE, col = "green", lwd = 2)

legend(
  "bottomright",
  legend = c(
    paste0("GLM: ", round(auc(roc_glm),3)),
    paste0("LASSO: ", round(auc(roc_lasso),3)),
    paste0("Ridge: ", round(auc(roc_ridge),3)),
    paste0("Elastic Net: ", round(auc(roc_elastic),3))
  ),
  col = c(
    "black",
    "blue",
    "red",
    "green"
  ),
  lwd = 2
)

dev.off()




# =============================================================================
# 3. Non-Linearity Assessment for GAM
# =============================================================================

# Before fitting a GAM, assess whether the relationship between
# predictors and the outcome is non-linear. Evidence of non-linearity
# supports the use of smooth functions in GAM.

# =============================================================================
# 3.1 Residuals vs Fitted Values (Global Check)
# =============================================================================

res_glm <- residuals(
  glm_base,
  type = "deviance"
)

fit_glm <- fitted(glm_base)

plot(
  fit_glm,
  res_glm,
  xlab = "Fitted Values",
  ylab = "Deviance Residuals",
  main = "Residuals vs Fitted (Baseline GLM)"
)

abline(
  h = 0,
  col = "red",
  lwd = 2
)

# =============================================================================
# 3.2 Predictor vs Residual Plots
# =============================================================================

continuous_predictors <- c(
  "UV_Index",
  "Wind_2M",
  "Temp",
  "Precipitation",
  "Rel_Humidity",
  "Pressure",
  "Wind_10M",
  "Wind_Dir_10M"
)

par(mfrow = c(2, 4))

for (v in continuous_predictors) {
  
  plot(
    train_df[[v]],
    res_glm,
    xlab = v,
    ylab = "Deviance Residuals",
    main = paste("Residuals vs", v)
  )
  
  lines(
    lowess(
      train_df[[v]],
      res_glm
    ),
    col = "blue",
    lwd = 2
  )
}

par(mfrow = c(1, 1))

# =============================================================================
# 3.3 Component + Residual (Partial Residual) Plots
# =============================================================================

library(car)

crPlots(glm_base)






# =============================================================================
# 4. Generalized Additive Models (GAM)
# =============================================================================

library(mgcv)

# =============================================================================
# 4.1 Baseline GAM
# =============================================================================

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

# =============================================================================
# 4.2 GAM-LASSO (Variable Selection)
# =============================================================================

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
  select = TRUE,
  gamma = 1.4
)

summary(gam_lasso)

gam.check(gam_lasso)

# =============================================================================
# 4.3 GAM-Ridge (Strong Smoothing)
# =============================================================================

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
  gamma = 2.0
)

summary(gam_ridge)

gam.check(gam_ridge)

# =============================================================================
# 4.4 GAM-Elastic Net
# =============================================================================

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
  select = TRUE,
  gamma = 1.8
)

summary(gam_enet)

gam.check(gam_enet)

# =============================================================================
# Model Evaluation
# =============================================================================

library(caret)
library(pROC)

compute_metrics <- function(
    model,
    test_df,
    model_name,
    threshold = 0.5
) {
  
  preds_prob <- predict(
    model,
    newdata = test_df,
    type = "response"
  )
  
  preds_class <- ifelse(
    preds_prob >= threshold,
    "Yes",
    "No"
  )
  
  preds_class <- factor(
    preds_class,
    levels = c("No", "Yes")
  )
  
  cm <- confusionMatrix(
    preds_class,
    test_df$Status,
    positive = "Yes"
  )
  
  auc_value <- roc(
    test_df$Status,
    preds_prob
  )$auc
  
  list(
    Model = model_name,
    Accuracy = cm$overall["Accuracy"],
    Precision = cm$byClass["Precision"],
    Recall = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    AUC = auc_value,
    ConfusionMatrix = cm$table
  )
}

# =============================================================================
# Evaluate All Models
# =============================================================================

m1 <- compute_metrics(
  gam_base,
  test_df,
  "GAM Baseline"
)

m2 <- compute_metrics(
  gam_lasso,
  test_df,
  "GAM LASSO"
)

m3 <- compute_metrics(
  gam_ridge,
  test_df,
  "GAM Ridge"
)

m4 <- compute_metrics(
  gam_enet,
  test_df,
  "GAM Elastic Net"
)

results_gam <- data.frame(
  Model = c(
    m1$Model,
    m2$Model,
    m3$Model,
    m4$Model
  ),
  Accuracy = c(
    m1$Accuracy,
    m2$Accuracy,
    m3$Accuracy,
    m4$Accuracy
  ),
  Precision = c(
    m1$Precision,
    m2$Precision,
    m3$Precision,
    m4$Precision
  ),
  Recall = c(
    m1$Recall,
    m2$Recall,
    m3$Recall,
    m4$Recall
  ),
  Specificity = c(
    m1$Specificity,
    m2$Specificity,
    m3$Specificity,
    m4$Specificity
  ),
  AUC = c(
    m1$AUC,
    m2$AUC,
    m3$AUC,
    m4$AUC
  )
)

print(results_gam)

# Save Results

write.csv(
  results_gam,
  file.path(
    output_dir,
    "GAM_Model_Performance.csv"
  ),
  row.names = FALSE
)

# =============================================================================
# Confusion Matrices
# =============================================================================

m1$ConfusionMatrix
m2$ConfusionMatrix
m3$ConfusionMatrix
m4$ConfusionMatrix

# =============================================================================
# ROC Curve Comparison
# =============================================================================

preds_prob_list <- list(
  "GAM Baseline" = predict(
    gam_base,
    test_df,
    type = "response"
  ),
  
  "GAM LASSO" = predict(
    gam_lasso,
    test_df,
    type = "response"
  ),
  
  "GAM Ridge" = predict(
    gam_ridge,
    test_df,
    type = "response"
  ),
  
  "GAM Elastic Net" = predict(
    gam_enet,
    test_df,
    type = "response"
  )
)

actual <- test_df$Status

roc_list <- lapply(
  preds_prob_list,
  function(x) roc(actual, x)
)

auc_values <- sapply(
  roc_list,
  function(x) round(auc(x), 3)
)

png(
  file.path(
    output_dir,
    "ROC_All_GAM_Models.png"
  ),
  width = 1200,
  height = 900
)

plot(
  roc_list[[1]],
  col = "red",
  lwd = 2,
  main = "ROC Curves for GAM Models"
)

plot(roc_list[[2]],
     add = TRUE,
     col = "blue",
     lwd = 2)

plot(roc_list[[3]],
     add = TRUE,
     col = "green",
     lwd = 2)

plot(roc_list[[4]],
     add = TRUE,
     col = "purple",
     lwd = 2)

legend(
  "bottomright",
  legend = paste0(
    names(roc_list),
    " (AUC = ",
    auc_values,
    ")"
  ),
  col = c(
    "red",
    "blue",
    "green",
    "purple"
  ),
  lwd = 2
)

dev.off()

# =============================================================================
# Best GAM Model ROC Curve
# =============================================================================

roc_base <- roc(
  test_df$Status,
  predict(
    gam_base,
    test_df,
    type = "response"
  )
)

auc_base <- round(
  auc(roc_base),
  3
)

png(
  file.path(
    output_dir,
    "ROC_GAM_Baseline.png"
  ),
  width = 1200,
  height = 900
)

plot(
  roc_base,
  col = "red",
  lwd = 2,
  legacy.axes = TRUE,
  main = paste0(
    "ROC Curve - GAM Baseline (AUC = ",
    auc_base,
    ")"
  )
)

legend(
  "bottomright",
  legend = paste0(
    "GAM Baseline (AUC = ",
    auc_base,
    ")"
  ),
  col = "red",
  lwd = 2
)

dev.off()





#========================================================================================================================#
#---------------------------------- Feature Importance Analysis : Best GAM Model ----------------------------------------#
#========================================================================================================================#

# =============================================================================
# 5. Feature Importance Analysis
# =============================================================================

# =============================================================================
# Best Performing GAM Model
# =============================================================================

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

# =============================================================================
# Custom Variable Labels
# =============================================================================

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

# =============================================================================
# Extract Smooth-Term Contributions
# =============================================================================

shap_smooth <- as.data.frame(predict(gam_base, type = "terms"))
shap_smooth <- shap_smooth[, !grepl("Wind_2M|Precipitation|Season", names(shap_smooth))]
names(shap_smooth) <- gsub("s\\(|\\)", "", names(shap_smooth))

# =============================================================================
# Extract Linear-Term Contributions
# =============================================================================

coef_all <- coef(gam_base)

shap_linear <- data.frame(
  Wind_2M       = train_df$Wind_2M * coef_all["Wind_2M"],
  Precipitation = train_df$Precipitation * coef_all["Precipitation"]
)

# =============================================================================
# Extract Seasonal Contributions
# =============================================================================

season_mm <- model.matrix(~ Season, data = train_df)

season_coef_all <- coef_all[grep("Season", names(coef_all))]

shap_season <- as.data.frame(
  season_mm[, names(season_coef_all), drop = FALSE] %*%
    diag(season_coef_all)
)

names(shap_season) <- paste0("SHAP_", names(season_coef_all))

# =============================================================================
# Calculate Feature Importance
# =============================================================================

shap_full <- cbind(
  shap_smooth,
  shap_linear,
  shap_season
)

imp <- colMeans(abs(shap_full))

imp_df <- data.frame(
  Variable = names(imp),
  Importance = imp
)

imp_df <- imp_df[
  order(
    imp_df$Importance,
    decreasing = TRUE
  ),
]

# =============================================================================
# Feature Importance Plot
# =============================================================================

library(ggplot2)

ggplot(imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue", width = 0.6) +
  coord_flip() +
  labs(
    title = "Feature Importance from GAM",
    x = "Variables",
    y = "Mean Absolute Contribution"
  ) +
  scale_x_discrete(labels = custom_names) +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size = 13, face = "bold"),
    axis.text = element_text(size = 11, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
  )

# =============================================================================
# Save Figure
# =============================================================================

ggsave(
  filename = "C:/Users/HPC/Downloads/HPAI_Feature_Importance.png",
  width = 8,
  height = 8,
  dpi = 600,
  bg = "white"
)




#========================================================================================================================#
#-------------------------------- Point-Based Risk Mapping Using Best GAM Model -----------------------------------------#
#========================================================================================================================#

# =============================================================================
# 11. Point-Based Risk Mapping
# =============================================================================

# =============================================================================
# Load Spatial and Climate Data
# =============================================================================

# Data
setindia_grid <- st_read("C:/Users/HPC/Desktop/Virocon/0.4_Inida_grid/India_grid_0.4_shp.shp")
raw_df <- read.csv("C:/Users/HPC/Downloads/HPAI_Climate_data.csv")

# =============================================================================
# Load Required Libraries
# =============================================================================

library(mgcv)
library(sf)
library(ggplot2)
library(ggspatial)
library(mgcv)

# =============================================================================
# Load Best Performing GAM Model
# =============================================================================

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

# =============================================================================
# Load India Grid Shapefile
# =============================================================================

india_grid <- st_read("C:/Users/HPC/Desktop/Virocon/0.4_Inida_grid/India_grid_0.4_shp.shp")
india_grid$ST_NM <- toupper(india_grid$ST_NM)

# =============================================================================
# Load Raw Climate Data
# =============================================================================

raw_df <- read.csv("C:/Users/HPC/Downloads/HPAI_Climate_data.csv")

# =============================================================================
# Standardize Variable Names
# =============================================================================

colnames(raw_df)[names(raw_df) == "UV_Index..W.m.2.x.40."] <- "UV_Index"
colnames(raw_df)[names(raw_df) == "Wind_Speed.at.2.Meters..m.s."] <- "Wind_2M"
colnames(raw_df)[names(raw_df) == "Temperature.at.2.Meters..C."] <- "Temp"
colnames(raw_df)[names(raw_df) == "Relative.Humidity.at.2.Meters...."] <- "Rel_Humidity"
colnames(raw_df)[names(raw_df) == "Precipitation.Corrected..mm.day."] <- "Precipitation"
colnames(raw_df)[names(raw_df) == "Surface.Pressure..kPa."] <- "Pressure"
colnames(raw_df)[names(raw_df) == "Wind.Speed.at.10.Meters..m.s."] <- "Wind_10M"
colnames(raw_df)[names(raw_df) == "Wind.Direction.at.10.Meters..Degrees."] <- "Wind_Dir_10M"

# =============================================================================
# Select Predictor Variables
# =============================================================================

raw_df <- raw_df[, c(
  "UV_Index","Temp","Rel_Humidity","Pressure","Wind_10M",
  "Wind_2M","Precipitation","Season","Longitude","Latitude"
)]

# =============================================================================
# Missing Value Imputation
# =============================================================================

num_cols <- sapply(raw_df, is.numeric)

raw_df[num_cols] <- lapply(
  raw_df[num_cols],
  function(x) {
    x[is.na(x)] <- median(x, na.rm = TRUE)
    x
  }
)

# =============================================================================
# Match Factor Levels
# =============================================================================

raw_df$Season <- factor(
  raw_df$Season,
  levels = levels(train_df$Season)
)

# =============================================================================
# Predict Outbreak Probability
# =============================================================================

raw_df$pred_prob <- predict(
  gam_base,
  newdata = raw_df,
  type = "response"
)

summary(raw_df$pred_prob)

# =============================================================================
# Export Predicted Probabilities
# =============================================================================

pred_out <- raw_df[, c(
  "Latitude",
  "Longitude",
  "pred_prob"
)]

write.csv(
  pred_out,
  "C:/Users/HPC/Downloads/GAM_predicted_probability_latlong.csv",
  row.names = FALSE
)

# =============================================================================
# Create Risk Categories
# =============================================================================

q <- quantile(
  raw_df$pred_prob,
  probs = c(0, 0.2, 0.6, 0.9, 1),
  na.rm = TRUE
)

raw_df$risk_class <- cut(
  raw_df$pred_prob,
  breaks = q,
  labels = c(
    "Low",
    "Medium-Low",
    "Medium-High",
    "High"
  ),
  include.lowest = TRUE
)

table(raw_df$risk_class)

# =============================================================================
# Convert to Spatial Object
# =============================================================================

raw_sf <- st_as_sf(
  raw_df,
  coords = c("Longitude", "Latitude"),
  crs = 4326
)

if (st_crs(raw_sf) != st_crs(india_grid)) {
  raw_sf <- st_transform(
    raw_sf,
    st_crs(india_grid)
  )
}

# =============================================================================
# Point-Based HPAI Risk Map
# =============================================================================

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
    plot.title        = element_text(
      hjust = 0.5,
      face = "bold"
    ),
    legend.position   = "right",
    panel.grid        = element_blank(),
    axis.title        = element_blank(),
    axis.text         = element_blank(),
    axis.ticks        = element_blank()
  )

# =============================================================================
# Save Risk Map
# =============================================================================

ggsave(
  filename = "C:/Users/HPC/Downloads/HPAI_Outbreak_Risk_Map.png",
  width = 10,
  height = 12,
  dpi = 600,
  bg = "white"
)


