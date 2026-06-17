

# =============================================================================
# 1. Spatial Autocorrelation Analysis
# =============================================================================

# =============================================================================
# 1.1 Load Required Libraries
# =============================================================================

library(sf)
library(dplyr)
library(spdep)
library(tmap)
library(RColorBrewer)
library(mgcv)
library(writexl)

# =============================================================================
# 1.2 Load and Prepare Dataset
# =============================================================================

raw_df <- read.csv("C:\\Users\\HPC\\Downloads\\HPAI_Climate_data.csv")

head(raw_df)
names(raw_df)

# =============================================================================
# 12.3 State-wise Outbreak Summary
# =============================================================================

outbreaks_by_state <- raw_df %>%
  group_by(States) %>%
  summarise(
    outbreak_count = n_distinct(Event.ID)
  ) %>%
  arrange(desc(outbreak_count))

outbreaks_by_state

write_xlsx(
  outbreaks_by_state,
  path = "C:/Users/HPC/Downloads/outbreaks_by_state.xlsx"
)

# =============================================================================
# 1.4 Standardize Variable Names
# =============================================================================

colnames(raw_df)[colnames(raw_df) == "UV_Index..W.m.2.x.40."] <- "UV_Index"
colnames(raw_df)[colnames(raw_df) == "Wind_Speed.at.2.Meters..m.s."] <- "Wind_2M"
colnames(raw_df)[colnames(raw_df) == "Temperature.at.2.Meters..C."] <- "Temp"
colnames(raw_df)[colnames(raw_df) == "Relative.Humidity.at.2.Meters...."] <- "Rel_Humidity"
colnames(raw_df)[colnames(raw_df) == "Precipitation.Corrected..mm.day."] <- "Precipitation"
colnames(raw_df)[colnames(raw_df) == "Surface.Pressure..kPa."] <- "Pressure"
colnames(raw_df)[colnames(raw_df) == "Wind.Speed.at.10.Meters..m.s."] <- "Wind_10M"
colnames(raw_df)[colnames(raw_df) == "Wind.Direction.at.10.Meters..Degrees."] <- "Wind_Dir_10M"

# =============================================================================
# 1.5 Select Variables for Spatial Analysis
# =============================================================================

raw_df <- raw_df[, c(
  "UV_Index", "Wind_2M", "Temp", "Rel_Humidity",
  "Precipitation", "Pressure", "Wind_10M", "Wind_Dir_10M",
  "Season", "Status", "Longitude", "Latitude"
)]

# =============================================================================
# 1.6 Create Binary Outbreak Variable
# =============================================================================

raw_df$Status_bin <- ifelse(
  raw_df$Status == "Outbreak",
  1,
  0
)

# =============================================================================
# 1.7 Convert to Spatial Object
# =============================================================================

raw_sf <- st_as_sf(
  raw_df,
  coords = c("Longitude","Latitude"),
  crs = 4326
)

# =============================================================================
# 1.8 Global Moran's I Analysis
# =============================================================================

coords <- st_coordinates(raw_sf)

coords_jitter <- jitter(
  coords,
  amount = 0.0001
)

knn <- knearneigh(
  coords_jitter,
  k = 8
)

nb <- knn2nb(knn)

lw <- nb2listw(
  nb,
  style = "W",
  zero.policy = TRUE
)

moran_result <- moran.test(
  raw_sf$Status_bin,
  lw,
  zero.policy = TRUE
)

print(moran_result)

# =============================================================================
# 1.9 Local Moran's I (LISA)
# =============================================================================

local_moran <- localmoran(
  raw_sf$Status_bin,
  lw,
  zero.policy = TRUE
)

raw_sf$Ii   <- local_moran[,1]
raw_sf$Z_Ii <- local_moran[,4]

# =============================================================================
# 1.10 LISA Cluster Classification
# =============================================================================

q_low  <- quantile(
  raw_sf$Z_Ii,
  0.25,
  na.rm = TRUE
)

q_high <- quantile(
  raw_sf$Z_Ii,
  0.75,
  na.rm = TRUE
)

raw_sf$cluster <- "Not significant"

raw_sf$cluster[
  raw_sf$Status_bin == 1 &
    raw_sf$Z_Ii >= q_high
] <- "High-High"

raw_sf$cluster[
  raw_sf$Status_bin == 0 &
    raw_sf$Z_Ii <= q_low
] <- "Low-Low"

raw_sf$cluster[
  raw_sf$Status_bin == 0 &
    raw_sf$Z_Ii >= q_high
] <- "Low-High"

raw_sf$cluster[
  raw_sf$Status_bin == 1 &
    raw_sf$Z_Ii <= q_low
] <- "High-Low"

raw_sf$cluster[
  is.na(raw_sf$cluster)
] <- "Not significant"

raw_sf$cluster <- factor(
  raw_sf$cluster,
  levels = c(
    "High-High",
    "Low-Low",
    "Low-High",
    "High-Low",
    "Not significant"
  )
)

table(raw_sf$cluster)

# =============================================================================
# 1.11 District-Level LISA Mapping
# =============================================================================

# Load India Grid Shapefile

india_grid <- st_read(
  "C:\\Users\\HPC\\Desktop\\Virocon\\0.4_Inida_grid\\India_grid_0.4_shp.shp"
)

india_grid <- st_transform(
  india_grid,
  st_crs(raw_sf)
)

# Merge Grid Cells into Districts

india_districts <- india_grid %>%
  group_by(DISTRICT) %>%
  summarise()

# Spatial Join

points_joined <- st_join(
  raw_sf,
  india_districts,
  join = st_within
)

# =============================================================================
# 1.12 District-Level Cluster Aggregation
# =============================================================================

district_lisa <- points_joined %>%
  st_set_geometry(NULL) %>%
  group_by(DISTRICT) %>%
  summarise(
    HH = sum(cluster == "High-High"),
    LL = sum(cluster == "Low-Low"),
    HL = sum(cluster == "High-Low"),
    LH = sum(cluster == "Low-High")
  ) %>%
  rowwise() %>%
  mutate(
    max_val = max(c(HH, LL, HL, LH)),
    LISA_category = case_when(
      max_val == 0 ~ "No Outbreaks",
      HH == max_val ~ "High-High",
      LL == max_val ~ "Low-Low",
      HL == max_val ~ "High-Low",
      LH == max_val ~ "Low-High",
      TRUE ~ "Mixed"
    )
  ) %>%
  ungroup() %>%
  select(
    DISTRICT,
    LISA_category
  )

# =============================================================================
# 1.13 Join LISA Categories to Districts
# =============================================================================

india_districts <- left_join(
  india_districts,
  district_lisa,
  by = "DISTRICT"
)

india_districts$LISA_category[
  is.na(india_districts$LISA_category)
] <- "No Outbreaks"

# =============================================================================
# 1.14 Prepare Final LISA Categories
# =============================================================================

india_districts$LISA_category <- as.character(
  india_districts$LISA_category
)

india_districts$LISA_category[
  is.na(india_districts$LISA_category) |
    india_districts$LISA_category == "No Outbreaks"
] <- "No Outbreaks"

india_districts$LISA_category <- factor(
  india_districts$LISA_category,
  levels = c(
    "High-High",
    "Low-Low",
    "High-Low",
    "Low-High",
    "No Outbreaks"
  )
)

# =============================================================================
# 1.15 Define LISA Cluster Colors
# =============================================================================

my_colors <- c(
  "High-High"    = "#B2182B",
  "Low-Low"      = "#2166AC",
  "High-Low"     = "#FFC067",
  "Low-High"     = "#DD7F82",
  "No Outbreaks" = "lightblue"
)

# =============================================================================
# 1.16 District-Level LISA Cluster Map
# =============================================================================

my_map <- tm_shape(india_districts) +
  tm_polygons(
    col = "LISA_category",
    palette = my_colors,
    colorNA = "white",
    border.col = "lightblue",
    lwd = 0.8,
    legend.show = FALSE
  ) +
  
  tm_add_legend(
    type = "fill",
    labels = c(
      "High–High: Hotspot district",
      "Low–Low: Coldspot district",
      "High–Low: High-risk outlier",
      "Low–High: Low-risk outlier",
      "No Outbreaks"
    ),
    col = my_colors,
    title = "LISA Cluster Categories"
  ) +
  
  tm_layout(
    main.title = "District-Level LISA Cluster Map of HPAI Outbreaks",
    main.title.size = 2.2,
    main.title.fontface = "bold",
    legend.outside = TRUE,
    legend.outside.position = "right",
    legend.outside.size = 0.25,
    legend.title.size = 1.0,
    legend.text.size = 0.8,
    frame = FALSE
  ) +
  
  tm_compass(
    position = c("left", "top")
  )

my_map

# =============================================================================
# 1.17 Save District-Level LISA Map
# =============================================================================

tmap_save(
  my_map,
  filename = "C:/Users/HPC/Downloads/HPAI_LISA_map.png",
  width = 10,
  height = 8,
  dpi = 600
)






# =============================================================================
# 2. Hotspot and High-Risk District Identification
# =============================================================================

# =============================================================================
# 2.1 Extract High-High LISA Hotspot Districts
# =============================================================================

# Extract High-High hotspot districts
hotspot_districts <- india_districts %>%
  st_set_geometry(NULL) %>%
  filter(LISA_category == "High-High") %>%
  distinct(DISTRICT)

# Write to CSV
write.csv(
  hotspot_districts,
  "C:/Users/HPC/Downloads/High_High_LISA_Hotspot_Districts.csv",
  row.names = FALSE
)

hotspot_names <- hotspot_districts$DISTRICT
hotspot_names

nrow(hotspot_districts)

# =============================================================================
# 2.2 Extract High-Risk Districts (High-High + High-Low)
# =============================================================================

high_risk_districts <- india_districts %>%
  st_set_geometry(NULL) %>%
  filter(LISA_category %in% c("High-High", "High-Low")) %>%
  distinct(DISTRICT, LISA_category)

high_risk_districts

table(india_districts$LISA_category)

# =============================================================================
# 2.3 Identify High-Risk Prediction Points
# =============================================================================

library(sf)

df <- read.csv(
  "C:\\Users\\HPC\\Downloads\\GAM_predicted_probability_latlong.csv"
)

head(df)

summary(df$pred_prob)

# Filter rows with predicted probability > 0.90
high_risk <- df[df$pred_prob > 0.9, ]

head(high_risk)

# Save to CSV
write.csv(
  high_risk,
  "C:/Users/HPC/Downloads/high_risk_points.csv",
  row.names = FALSE
)

# =============================================================================
# 2.4 Prepare High-Risk District Dataset
# =============================================================================

# Note:
# High-risk points were extracted from the GAM prediction output.
# Duplicate latitude/longitude locations were removed.
# Corresponding district and state names were manually assigned.
# File used: High_Risk_District_data.csv

library(dplyr)
library(stringr)

# Grid shapefile
india_grid <- st_read(
  "C:/Users/HPC/Desktop/Virocon/0.4_Inida_grid/India_grid_0.4_shp.shp",
  quiet = TRUE
)

india_grid <- india_grid %>%
  mutate(
    DISTRICT_clean = str_to_lower(
      str_trim(DISTRICT)
    )
  )

# Load district dataset
df <- read.csv(
  "C:\\Users\\HPC\\Downloads\\High_Risk_District_data.csv"
)

head(df)

unique(df$District)

df <- df %>%
  mutate(
    DISTRICT_clean = str_to_lower(
      str_trim(District)
    )
  )

# =============================================================================
# 2.5 District Name Harmonization
# =============================================================================

district_fix <- c(
  
  # Telangana / Andhra Pradesh / Karnataka
  "hydrebad" = "hyderabad",
  "gilbarga" = "kalaburagi",
  "ballari" = "bellary",
  
  # Maharashtra
  "navi mumbai" = "thane",
  "buldana" = "buldhana",
  "ahmednagar" = "ahmadnagar",
  
  # Kerala
  "kozhi kode" = "kozhikode",
  
  # Gujarat
  "ahmedabad" = "ahmadabad",
  
  # Odisha / West Bengal
  "purba bardhaman" = "purba bardhaman",
  "birbhum" = "birbhum",
  "hooghly" = "hugli",
  "jagatsinghapur" = "jagatsinghapur",
  "cooch behar" = "koch bihar",
  
  # Haryana
  "panchakula" = "panchkula",
  
  # Jammu & Kashmir
  "baramulla" = "baramula",
  
  # Meghalaya
  "west khasi hills" = "west khasi hills",
  
  # Delhi
  "new delhi" = "new delhi"
)

df <- df %>%
  mutate(
    DISTRICT_final = dplyr::recode(
      DISTRICT_clean,
      !!!district_fix,
      .default = DISTRICT_clean
    )
  )

setdiff(
  df$DISTRICT_final,
  india_grid$DISTRICT_clean
)

# =============================================================================
# 2.6 Final District Matching Corrections
# =============================================================================

district_fix_final <- c(
  "maysuru" = "mysore",
  "darjeeling" = "darjiling",
  "malda" = "maldah",
  "bribhum" = "birbhum",
  "howrah" = "haora",
  "buldhana" = "buldana",
  "palghar" = "thane",
  "kalaburagi" = "gulbarga",
  "purba bardhaman" = "bardhaman"
)

df <- df %>%
  mutate(
    DISTRICT_final = dplyr::recode(
      DISTRICT_final,
      !!!district_fix_final,
      .default = DISTRICT_final
    )
  )

setdiff(
  df$DISTRICT_final,
  india_grid$DISTRICT_clean
)

# =============================================================================
# 2.7 Create District and State Boundaries
# =============================================================================

india_districts <- india_grid %>%
  group_by(DISTRICT_clean) %>%
  summarise(
    geometry = st_union(geometry),
    .groups = "drop"
  )

highlight_districts <- india_districts %>%
  filter(
    DISTRICT_clean %in% df$DISTRICT_final
  )

state_boundaries <- india_grid %>%
  group_by(ST_NM) %>%
  summarise(
    geometry = st_union(geometry),
    .groups = "drop"
  )

# =============================================================================
# 2.8 High-Risk District Map
# =============================================================================

library(ggplot2)
library(ggspatial)

p <- ggplot() +
  geom_sf(
    data = india_districts,
    fill = "grey90",
    color = "white",
    size = 0.1
  ) +
  geom_sf(
    data = highlight_districts,
    fill = "#E63946",
    color = "#8B0000",
    size = 0.35
  ) +
  geom_sf(
    data = state_boundaries,
    fill = NA,
    color = "#2C3E50",
    size = 0.7
  ) +
  annotation_north_arrow(
    location = "tr",
    which_north = "true",
    style = north_arrow_fancy_orienteering
  ) +
  labs(
    title = "High-Risk Districts in India"
  ) +
  theme_void() +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      size = 18,
      face = "bold"
    )
  )

# =============================================================================
# 2.9 Save High-Risk District Map
# =============================================================================

ggsave(
  filename = "C:/Users/HPC/Downloads/High_risk_district.png",
  plot = p,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"
)