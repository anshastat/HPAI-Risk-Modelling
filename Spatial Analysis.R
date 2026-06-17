========================================================================== Spatial Analysis ==================================================================

# R code


#------------------------ spatial autocorrelation -------------------#


#  Load libraries
library(sf)
library(dplyr)
library(spdep)
library(tmap)
library(RColorBrewer)
library(mgcv)
library(writexl)


#  Load and prepare raw dataset
raw_df <- read.csv("C:\\Users\\HPC\\Downloads\\HPAI_Climate_data.csv")
head(raw_df)
names(raw_df)

# outbreaks count per states

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

# Rename columns
colnames(raw_df)[colnames(raw_df) == "UV_Index..W.m.2.x.40."] <- "UV_Index"
colnames(raw_df)[colnames(raw_df) == "Wind_Speed.at.2.Meters..m.s."] <- "Wind_2M"
colnames(raw_df)[colnames(raw_df) == "Temperature.at.2.Meters..C."] <- "Temp"
colnames(raw_df)[colnames(raw_df) == "Relative.Humidity.at.2.Meters...."] <- "Rel_Humidity"
colnames(raw_df)[colnames(raw_df) == "Precipitation.Corrected..mm.day."] <- "Precipitation"
colnames(raw_df)[colnames(raw_df) == "Surface.Pressure..kPa."] <- "Pressure"
colnames(raw_df)[colnames(raw_df) == "Wind.Speed.at.10.Meters..m.s."] <- "Wind_10M"
colnames(raw_df)[colnames(raw_df) == "Wind.Direction.at.10.Meters..Degrees."] <- "Wind_Dir_10M"

# Keep only selected variables
raw_df <- raw_df[, c(
  "UV_Index", "Wind_2M", "Temp", "Rel_Humidity",
  "Precipitation", "Pressure", "Wind_10M", "Wind_Dir_10M",
  "Season", "Status", "Longitude", "Latitude"
)]

# Create binary outbreak status
raw_df$Status_bin <- ifelse(raw_df$Status == "Outbreak", 1, 0)

# Convert to sf
raw_sf <- st_as_sf(raw_df, coords = c("Longitude","Latitude"), crs = 4326)

#  Global Moran's I
coords <- st_coordinates(raw_sf)
coords_jitter <- jitter(coords, amount = 0.0001) # avoid duplicates
knn <- knearneigh(coords_jitter, k = 8)
nb <- knn2nb(knn)
lw <- nb2listw(nb, style = "W", zero.policy = TRUE)

moran_result <- moran.test(raw_sf$Status_bin, lw, zero.policy = TRUE)
print(moran_result)  

#  Local Moran's I (LISA)
local_moran <- localmoran(raw_sf$Status_bin, lw, zero.policy = TRUE)
raw_sf$Ii   <- local_moran[,1]  # Local Moran's I
raw_sf$Z_Ii <- local_moran[,4]  # Z-score

# Compute quantiles
q_low  <- quantile(raw_sf$Z_Ii, 0.25, na.rm = TRUE)
q_high <- quantile(raw_sf$Z_Ii, 0.75, na.rm = TRUE)

# Assign LISA clusters
raw_sf$cluster <- "Not significant"
raw_sf$cluster[raw_sf$Status_bin == 1 & raw_sf$Z_Ii >= q_high] <- "High-High"
raw_sf$cluster[raw_sf$Status_bin == 0 & raw_sf$Z_Ii <= q_low]  <- "Low-Low"
raw_sf$cluster[raw_sf$Status_bin == 0 & raw_sf$Z_Ii >= q_high] <- "Low-High"
raw_sf$cluster[raw_sf$Status_bin == 1 & raw_sf$Z_Ii <= q_low]  <- "High-Low"
raw_sf$cluster[is.na(raw_sf$cluster)] <- "Not significant"
raw_sf$cluster <- factor(raw_sf$cluster, levels = c("High-High", "Low-Low", "Low-High", "High-Low", "Not significant"))

# Count of each LISA cluster
table(raw_sf$cluster)


#  ------------------ District-level LISA map ---------------------#

# Load India grid shapefile
india_grid <- st_read("C:\\Users\\HPC\\Desktop\\Virocon\\0.4_Inida_grid\\India_grid_0.4_shp.shp")
india_grid <- st_transform(india_grid, st_crs(raw_sf))

# Merge grids into districts
india_districts <- india_grid %>% group_by(DISTRICT) %>% summarise()

# Spatial join: assign points to districts
points_joined <- st_join(raw_sf, india_districts, join = st_within)

# Aggregate LISA clusters per district
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
  select(DISTRICT, LISA_category)

# Join back to districts
india_districts <- left_join(india_districts, district_lisa, by = "DISTRICT")

# Fill NA categories for districts with no points
india_districts$LISA_category[is.na(india_districts$LISA_category)] <- "No Outbreaks"


#  Plot district-level LISA map 

# Merge NA and No Outbreaks
india_districts$LISA_category <- as.character(india_districts$LISA_category)
india_districts$LISA_category[is.na(india_districts$LISA_category) | 
                                india_districts$LISA_category == "No Outbreaks"] <- "No Outbreaks"
india_districts$LISA_category <- factor(india_districts$LISA_category, 
                                        levels = c("High-High", "Low-Low", "High-Low", "Low-High", "No Outbreaks"))




# Define colors
my_colors <- c(
  "High-High"    = "#B2182B",
  "Low-Low"      = "#2166AC",
  "High-Low"     = "#FFC067",
  "Low-High"     = "#DD7F82",
  "No Outbreaks" = "lightblue"
)

# Plot map
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
  
  tm_compass(position = c("left", "top"))

my_map


# Save map 
tmap_save(
  my_map,
  filename = "C:/Users/HPC/Downloads/HPAI_LISA_map.png",
  width = 10,
  height = 8,
  dpi = 600
)



#------------------ Hotspot district name ------------------------


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

high_risk_districts <- india_districts %>%
  st_set_geometry(NULL) %>%
  filter(LISA_category %in% c("High-High", "High-Low")) %>%
  distinct(DISTRICT, LISA_category)

high_risk_districts

table(india_districts$LISA_category)

#------------------------------------------------------------------# High Risk Districts In india -------------------------------------------------------------#

library(sf)
df = read.csv("C:\\Users\\HPC\\Downloads\\GAM_predicted_probability_latlong.csv")
head(df)
summary(df$pred_prob)

# Filter rows with predicted probability > 0.75
high_risk <- df[df$pred_prob > 0.9, ]

# Check first few rows
head(high_risk)

# Save to CSV
write.csv(
  high_risk,
  "C:/Users/HPC/Downloads/high_risk_points.csv",
  row.names = FALSE
)


# Note =  here I used high risk points data then remove duplicate lat/long then using unique lat/long I mentioned there corresponding district and state names
# file name = High_Risk_District_data.csv


library(dplyr)
library(stringr)

# Grid shapefile
india_grid <- st_read(
  "C:/Users/HPC/Desktop/Virocon/0.4_Inida_grid/India_grid_0.4_shp.shp",
  quiet = TRUE
)

india_grid <- india_grid %>%
  mutate(DISTRICT_clean = str_to_lower(str_trim(DISTRICT)))

# CSV
df <- read.csv("C:\\Users\\HPC\\Downloads\\High_Risk_District_data.csv")
head(df)
unique(df$District)

df <- df %>%
  mutate(DISTRICT_clean = str_to_lower(str_trim(District)))


district_fix <- c(
  # Telangana / AP / Karnataka
  "hydrebad"                  = "hyderabad",
  "gilbarga"                  = "kalaburagi",
  "ballari"                   = "bellary",
  
  # Maharashtra
  "navi mumbai"               = "thane",
  "buldana"                   = "buldhana",
  "ahmednagar"                = "ahmadnagar",
  
  # Kerala
  "kozhi kode"                = "kozhikode",
  
  # Gujarat
  "ahmedabad"                 = "ahmadabad",
  
  # Odisha / WB
  "purba bardhaman"           = "purba bardhaman",
  "birbhum"                   = "birbhum",
  "hooghly"                   = "hugli",
  "jagatsinghapur"            = "jagatsinghapur",
  "cooch behar"               = "koch bihar",
  
  # Haryana
  "panchakula"                = "panchkula",
  
  # J&K
  "baramulla"                 = "baramula",
  
  # Meghalaya
  "west khasi hills"          = "west khasi hills",
  
  # Delhi
  "new delhi"                 = "new delhi"
  
  
)


df <- df %>%
  mutate(
    DISTRICT_final = dplyr::recode(
      DISTRICT_clean,
      !!!district_fix,
      .default = DISTRICT_clean
    )
  )

setdiff(df$DISTRICT_final, india_grid$DISTRICT_clean)


district_fix_final <- c(
  "maysuru"     = "mysore",
  "darjeeling"  = "darjiling",
  "malda"       = "maldah",
  "bribhum"     = "birbhum",
  "howrah"      = "haora",
  "buldhana"    = "buldana",
  "palghar"         = "thane",
  "kalaburagi"      = "gulbarga",
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


setdiff(df$DISTRICT_final, india_grid$DISTRICT_clean)


# District polygons
india_districts <- india_grid %>%
  group_by(DISTRICT_clean) %>%
  summarise(geometry = st_union(geometry), .groups = "drop")

# Highlighted districts
highlight_districts <- india_districts %>%
  filter(DISTRICT_clean %in% df$DISTRICT_final)

# State boundaries
state_boundaries <- india_grid %>%
  group_by(ST_NM) %>%
  summarise(geometry = st_union(geometry), .groups = "drop")

# Plot
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
  labs(title = "High-Risk Districts in India") +
  theme_void() +
  theme(
    plot.title = element_text(
      hjust = 0.5,
      size = 18,
      face = "bold"
    )
  )

ggsave(
  filename = "C:/Users/HPC/Downloads/High_risk_district.png",
  plot = p,
  width = 10,
  height = 10,
  dpi = 600,
  bg = "white"
)
