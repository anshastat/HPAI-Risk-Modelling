#==================================================================== Psudo data generation ============================================================#


# =========================
# Download WorldClim data
# =========================
dir.create("data/worldclim", recursive = TRUE, showWarnings = FALSE)

wc_bio <- geodata::worldclim_global(
  var  = "bio",
  res  = 2.5,   # 5KM
  path = "data/worldclim"
)

# =========================
# Load India grid boundary
# =========================
india_grid <- st_read(
  "C:/Users/HPC/Desktop/Virocon/0.4_Inida_grid/India_grid_0.4_shp.shp",
  quiet = TRUE
)

india_grid <- st_make_valid(india_grid)
india_grid <- st_transform(india_grid, crs(wc_bio))
india_vect <- vect(india_grid)

# Crop & mask climate data
wc_bio_india <- crop(wc_bio, india_vect)
wc_bio_india <- mask(wc_bio_india, india_vect)



#==========================
# Presence Data Processing
# =========================

presence_df <- read.csv("C:/Users/HPC/Downloads/unique_latlong_district.csv")
names(presence_df)

presence_sf <- st_as_sf(
  presence_df,
  coords = c("Longitude", "Latitude"),
  crs = 4326
) %>%
  st_transform(crs(wc_bio))

presence_vect <- vect(presence_sf)


presence_clim <- terra::extract(wc_bio_india, presence_vect)


presence_sf <- presence_sf %>%
  mutate(
    Longitude = st_coordinates(.)[,1],
    Latitude  = st_coordinates(.)[,2]
  )


presence_data <- cbind(
  st_drop_geometry(presence_sf),
  presence_clim
) %>%
  select(-ID) %>%     # remove terra ID
  drop_na()

table(presence_data$presence)
dim(presence_data)


write.csv(
  presence_data,
  "C:/Users/HPC/Downloads/presence_worldclim_data.csv",
  row.names = FALSE
)





#---------------------------------------- merge wetland dataset --------------------------------------------#


library(data.table)
library(sf)
library(dplyr)
library(nngeo)

# Load waterways (all zones)
load_waterways <- function(shp_path){
  st_read(shp_path, layer = "gis_osm_waterways_free_1", quiet = TRUE)
}

western   <- load_waterways("C:/Users/HPC/Downloads/HPAI_Project/Waterways/western-zone-251211-free.shp")
central   <- load_waterways("C:/Users/HPC/Downloads/HPAI_Project/Waterways/central-zone-251211-free.shp")
eastern   <- load_waterways("C:/Users/HPC/Downloads/HPAI_Project/Waterways/eastern-zone-251210-free.shp")
northeast <- load_waterways("C:/Users/HPC/Downloads/HPAI_Project/Waterways/north-eastern-zone-251211-free.shp")
northern  <- load_waterways("C:/Users/HPC/Downloads/HPAI_Project/Waterways/northern-zone-251211-free.shp")
southern  <- load_waterways("C:/Users/HPC/Downloads/HPAI_Project/Waterways/southern-zone-251210-free.shp")

all_waterways <- rbind(western, central, eastern, northeast, northern, southern)

wetlands <- all_waterways %>%
  filter(fclass %in% c("river", "stream", "canal", "drain"))


model_data <- fread("C:/Users/HPC/Downloads/presence_worldclim_data.csv")

pa_sf <- st_as_sf(
  model_data,
  coords = c("Longitude", "Latitude"),
  crs = 4326
)

wetlands <- st_transform(wetlands, st_crs(pa_sf))

nn <- st_nn(pa_sf, wetlands, k = 1, returnDist = TRUE)

nearest_id <- sapply(nn$nn, `[`, 1)

pa_sf$dist_wetland_m  <- sapply(nn$dist, `[`, 1)
pa_sf$dist_wetland_km <- pa_sf$dist_wetland_m / 1000
pa_sf$nearest_water_type <- wetlands$fclass[nearest_id]


pa_sf <- pa_sf %>%
  mutate(
    wetland_1km = ifelse(dist_wetland_km <= 1, 1, 0),
    wetland_5km = ifelse(dist_wetland_km <= 5, 1, 0),
    wetland_dist_class = case_when(
      dist_wetland_km <= 1 ~ "0–1 km",
      dist_wetland_km <= 5 ~ "1–5 km",
      TRUE ~ ">5 km"
    )
  )

coords <- st_coordinates(pa_sf)
pa_sf$Longitude <- coords[,1]
pa_sf$Latitude  <- coords[,2]

final_df <- st_drop_geometry(pa_sf)

fwrite(
  final_df,
  "C:/Users/HPC/Downloads/presence_worldclim_wetland_data.csv"
)




#.............................................. Merge Poultry Density dataset ..........................................................#


df <- fread("C:/Users/HPC/Downloads/presence_worldclim_wetland_data.csv")

df_sf <- st_as_sf(df, coords = c("Longitude", "Latitude"),
                  crs = 4326, remove = FALSE)

# Load poultry shapefiles
chik_2015 <- st_read("C:/Users/HPC/Downloads/HPAI_Project/Polutry_density/chik_2015/district_poultry_density_2015.shp") %>%
  st_transform(st_crs(df_sf)) %>%
  rename(chicken_den_2015 = poultry_de)

chik_2010 <- st_read("C:/Users/HPC/Downloads/HPAI_Project/Polutry_density/chik_20210/district_poultry_density.shp") %>%
  st_transform(st_crs(df_sf)) %>%
  rename(chicken_den_2010 = poultry_de)

duck_2010 <- st_read("C:/Users/HPC/Downloads/HPAI_Project/Polutry_density/duck_2010/district_poultry_density_dk20210.shp") %>%
  st_transform(st_crs(df_sf)) %>%
  rename(duck_den_2010 = poultry_de)

duck_2015 <- st_read("C:/Users/HPC/Downloads/HPAI_Project/Polutry_density/duck_2015/district_poultry_density_dk20215.shp") %>%
  st_transform(st_crs(df_sf)) %>%
  rename(duck_den_2015 = poultry_de)

df_sf <- df_sf %>%
  st_join(chik_2015["chicken_den_2015"]) %>%
  st_join(chik_2010["chicken_den_2010"]) %>%
  st_join(duck_2010["duck_den_2010"]) %>%
  st_join(duck_2015["duck_den_2015"])

fwrite(st_drop_geometry(df_sf),
       "C:/Users/HPC/Downloads/presence_worldclim_wetland_poultry_data.csv")




# ...............................................Merge Additional Environmental Variables..........................................#

df_old <- fread("C:/Users/HPC/Downloads/presence_worldclim_wetland_poultry_data.csv")

dataset_list <- list(
  list(file = "C:/Users/HPC/Downloads/HPAI_Project/New_Data_Prep/da/India_LAI.csv", var_name = "LAI_2024"),
  list(file = "C:/Users/HPC/Downloads/HPAI_Project/New_Data_Prep/da/India_EVI.csv", var_name = "EVI_2024"),
  list(file = "C:/Users/HPC/Downloads/HPAI_Project/New_Data_Prep/da/India_LST.csv", var_name = "LST_2024"),
  list(file = "C:/Users/HPC/Downloads/HPAI_Project/New_Data_Prep/da/India_NDVI.csv", var_name = "NDVI_2024"),
  list(file = "C:/Users/HPC/Downloads/HPAI_Project/New_Data_Prep/da/India_PET.csv", var_name = "PET_2024"),
  list(file = "C:/Users/HPC/Downloads/HPAI_Project/New_Data_Prep/da/India_PotEvap_tavg.csv", var_name = "PotEvap_2024"),
  list(file = "C:/Users/HPC/Downloads/HPAI_Project/New_Data_Prep/da/India_Rainf_f_tavg.csv", var_name = "Rainf_2024"),
  list(file = "C:/Users/HPC/Downloads/HPAI_Project/New_Data_Prep/da/India_AirT_f_inst.csv", var_name = "AirT_2024"),
  list(file = "C:/Users/HPC/Downloads/HPAI_Project/New_Data_Prep/da/India_Wind_f_inst.csv", var_name = "Wind_2024"),
  list(file = "C:/Users/HPC/Downloads/HPAI_Project/New_Data_Prep/da/India_SurfP_f_inst.csv", var_name = "SurfP_2024"),
  list(file = "C:/Users/HPC/Downloads/HPAI_Project/New_Data_Prep/da/India_Sphqair_f_inst.csv", var_name = "Sphq_2024"),
  list(file = "C:/Users/HPC/Downloads/HPAI_Project/New_Data_Prep/da/India_SoilMoi0_10cm_inst.csv", var_name = "SoilMoist_2024")
)

process_env_var <- function(base_df, var_file, var_name){
  
  var_df <- fread(var_file)
  names(var_df) <- trimws(names(var_df))
  names(var_df) <- gsub("-", "_", names(var_df))
  
  cols_2024 <- grep("X?2024", names(var_df), value = TRUE)
  if(length(cols_2024) == 0) return(base_df)
  
  var_df <- var_df %>%
    rowwise() %>%
    mutate(tmp_median = median(c_across(all_of(cols_2024)), na.rm = TRUE)) %>%
    ungroup() %>%
    select(long, lat, tmp_median)
  
  names(var_df)[3] <- var_name
  
  base_sf <- st_as_sf(base_df, coords = c("Longitude", "Latitude"),
                      crs = 4326, remove = FALSE) %>%
    st_transform(3857) %>%
    mutate(point_id = row_number())
  
  var_sf <- st_as_sf(var_df, coords = c("long", "lat"),
                     crs = 4326) %>%
    st_transform(3857)
  
  base_sf <- base_sf %>%
    mutate(nn_id = st_nearest_feature(geometry, var_sf),
           nn_value = var_sf[[var_name]][nn_id]) %>%
    select(-nn_id)
  
  buffers <- st_buffer(base_sf, dist = 10000)
  var_in_buffer <- st_join(buffers, var_sf, join = st_intersects)
  
  var_buffer <- var_in_buffer %>%
    st_drop_geometry() %>%
    group_by(point_id) %>%
    summarise(buffer_value = median(.data[[var_name]], na.rm = TRUE),
              .groups = "drop")
  
  base_sf <- base_sf %>%
    left_join(var_buffer, by = "point_id") %>%
    mutate(!!var_name := ifelse(is.na(buffer_value), nn_value, buffer_value)) %>%
    select(-nn_value, -buffer_value)
  
  st_drop_geometry(base_sf)
}

for(ds in dataset_list){
  df_old <- process_env_var(df_old, ds$file, ds$var_name)
}

fwrite(df_old, "C:/Users/HPC/Downloads/HPAI_All_VARS_ML_Data.csv")




#============================= Insert Missing Values in poultry density using District avg ======================================#


df = read.csv("C:\\Users\\HPC\\Downloads\\HPAI_All_VARS_ML_Data.csv")

head(df)

library(dplyr)

cols_to_impute <- c("chicken_den_2015", "chicken_den_2010",
                    "duck_den_2010", "duck_den_2015")

df <- df %>%
  mutate(across(all_of(cols_to_impute), ~na_if(., -Inf)))


df <- df %>%
  group_by(District) %>%
  mutate(across(all_of(cols_to_impute),
                ~ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
  ungroup()


df <- df %>%
  mutate(across(all_of(cols_to_impute),
                ~ifelse(is.na(.), median(., na.rm = TRUE), .)))


summary(df[cols_to_impute])
colSums(is.na(df[cols_to_impute]))

# note - as poultry density variables are positively skewed so transform it to log transform

df <- df %>%
  mutate(across(all_of(cols_to_impute),
                ~log1p(.),
                .names = "{.col}_log"))


summary(df %>% select(ends_with("_log")))

# remove old poultry variables and add log variables

df <- df %>%
  select(-all_of(cols_to_impute))

names(df)

View(df)

# check the skewness
library(moments)

num_vars <- df %>%
  select(
    Annual.Mean.Temperature,           
    Mean.Diurnal.Range, Isothermality, Temperature.Seasonality,
    Max.Temperature.of.Warmest.Month, Min.Temperature.of.Coldest.Month,
    Temperature.Annual.Range, 
    Mean.Temperature.of.Wettest.Quarter, Mean.Temperature.of.Driest.Quarter,
    Mean.Temperature.of.Warmest.Quarter, Mean.Temperature.of.Coldest.Quarter,
    Annual.Precipitation, Precipitation.of.Wettest.Month, 
    Precipitation.of.Driest.Month, Precipitation.Seasonality,
    Precipitation.of.Wettest.Quarter, Precipitation.of.Driest.Quarter,
    Precipitation.of.Warmest.Quarter, Precipitation.of.Coldest.Quarter,
    LAI_2024, EVI_2024, LST_2024, NDVI_2024, PET_2024, PotEvap_2024, 
    Rainf_2024, AirT_2024, Wind_2024, SurfP_2024, Sphq_2024, SoilMoist_2024,
    chicken_den_2015_log, chicken_den_2010_log,
    duck_den_2010_log, duck_den_2015_log
  )

skew_values <- sapply(num_vars, skewness, na.rm = TRUE)

skew_df <- data.frame(
  Variable = names(skew_values),
  Skewness = skew_values
) %>%
  arrange(desc(Skewness))

skew_df


# final log transformation > 1


vars_log <- c(
  "Precipitation.of.Coldest.Quarter",
  "Precipitation.of.Driest.Month",
  "Precipitation.of.Warmest.Quarter",
  "Precipitation.of.Wettest.Month",
  "Precipitation.of.Driest.Quarter",
  "Precipitation.of.Wettest.Quarter",
  "Annual.Precipitation",
  "Rainf_2024",
  "LAI_2024",
  "PET_2024"
)

# log of variables

df <- df %>%
  mutate(across(all_of(vars_log),
                ~log1p(.),
                .names = "{.col}_log"))

df <- df %>%
  select(-all_of(vars_log))

names(df)

write.csv(df,
          "C:/Users/HPC/Downloads/All_vars_log.csv",
          row.names = FALSE)


#----------------------------------- create a data with unique districts --------------------------------------#


library(dplyr)

df <- read.csv("C:/Users/HPC/Downloads/All_vars_log.csv")

# Separate presence and background
df_bg <- df %>% filter(presence == 0)
df_pr <- df %>% filter(presence == 1)

# Mode function
get_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

df_bg_dist <- df_bg %>%
  group_by(District) %>%
  summarise(
    
    # Numeric → median (excluding coords & presence)
    across(
      .cols = setdiff(names(df_bg)[sapply(df_bg, is.numeric)],
                      c("Longitude", "Latitude", "presence")),
      .fns = ~median(.x, na.rm = TRUE)
    ),
    
    # Coordinates → median
    Longitude = median(Longitude, na.rm = TRUE),
    Latitude  = median(Latitude, na.rm = TRUE),
    
    # Categorical → mode
    nearest_water_type = get_mode(nearest_water_type),
    wetland_1km = get_mode(wetland_1km),
    wetland_5km = get_mode(wetland_5km),
    
    State = first(State),
    presence = 0,
    
    .groups = "drop"
  )

# Combine back
df_final <- bind_rows(df_pr, df_bg_dist)

# One background row per district
n_distinct(df_bg_dist$District)

# Class balance
table(df_final$presence)

# Check no duplicates for background districts
df_final %>%
  filter(presence == 0) %>%
  count(District) %>%
  filter(n > 1)


write.csv(df_final,
          "C:/Users/HPC/Downloads/df_sdm_presence_background_district.csv",
          row.names = FALSE)


