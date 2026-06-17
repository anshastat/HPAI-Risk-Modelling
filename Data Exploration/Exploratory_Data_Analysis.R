# ================================================================================
# ------------------Exploratory Data Analysis and Visualization-------------------
# ================================================================================


# =============================================================================
# 1. Load Required Libraries
# =============================================================================

library(dplyr)
library(ggplot2)
library(lubridate)
library(sf)
library(patchwork)
library(ggh4x)
library(viridis)

# =============================================================================
# 2. Define Project Paths
# =============================================================================

data_file <- "C:\\Users\\HPC\\Desktop\\Data Exploration\\EDA_Data.csv"

shapefile_path <- "data/shapefiles/Admin2.shp"

output_dir <- "outputs/EDA"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# =============================================================================
# 3. Load Data
# =============================================================================

df <- read.csv(data_file)

df$Observation_date <- as.Date(df$Observation_date)

# =============================================================================
# 4. Data Preparation
# =============================================================================

df$Season <- factor(
  df$Season,
  levels = c("Winter", "Summer", "Monsoon", "Autumn")
)

df <- df %>%
  mutate(
    Year = year(Observation_date)
  )

# =============================================================================
# 5. State-wise Distribution of HPAI Outbreaks
# =============================================================================

df_state <- df %>%
  filter(!is.na(States)) %>%
  group_by(States) %>%
  summarise(Total_Outbreaks = n(), .groups = "drop")

p1 <- ggplot(df_state, aes(x = reorder(States, -Total_Outbreaks), y = Total_Outbreaks)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  theme_minimal(base_size = 12) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 90, hjust = 1, face = "bold"),
    axis.text.y = element_text(face = "bold", size = 12),
    axis.title = element_text(face = "bold", size = 14),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16)  
  ) +
  xlab("State") +
  ylab("Number of Outbreaks") +
  ggtitle("State-wise Distribution of HPAI Outbreaks")

# Save
ggsave(
  file.path(output_dir, "Statewise_Outbreaks.tiff"),
  plot = p1,
  width = 10,
  height = 8,
  dpi = 600,
  device = "tiff",
  compression = "lzw"
)

# =============================================================================
# 6. Season-wise Distribution of HPAI Outbreaks
# =============================================================================

# Season-wise outbreaks
df_season <- df %>%
  filter(!is.na(Season)) %>%
  group_by(Season) %>%
  summarise(Season_Outbreaks = n(), .groups = "drop")

p2 <- ggplot(df_season, aes(x = Season, y = Season_Outbreaks, fill = Season)) +
  geom_bar(stat = "identity") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  scale_fill_manual(
    values = c(
      "Winter" = "#1f77b4",
      "Summer" = "#ff7f0e",
      "Monsoon" = "#2ca02c",
      "Autumn" = "#d62728"
    ),
    drop = FALSE
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid = element_blank(),
    axis.text = element_text(face = "bold", size = 12),
    axis.title = element_text(face = "bold", size = 14),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16)
  ) +
  xlab("Season") +
  ylab("Number of Outbreaks") +
  ggtitle("Season-wise Distribution of HPAI Outbreaks")

# save
ggsave(
  filename = file.path(output_folder, "Seasonwise_Outbreaks.tiff"),
  plot = p2,
  width = 10,
  height = 8,
  dpi = 600,
  device = "tiff",
  compression = "lzw"
)

# =============================================================================
# 7. Year-wise Distribution of HPAI Outbreaks
# =============================================================================

# Yearly outbreaks
df_year <- df %>%
  filter(!is.na(Observation_date)) %>%
  mutate(Year = year(Observation_date)) %>%
  group_by(Year) %>%
  summarise(Yearly_Outbreaks = n(), .groups = "drop")

p3 <- ggplot(df_year, aes(x = factor(Year), y = Yearly_Outbreaks)) +
  geom_bar(stat = "identity", fill = "tomato", width = 0.7) +
  scale_y_continuous(limits = c(0, 120)) +   # pdated
  theme_minimal(base_size = 14) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid = element_blank(),
    axis.text.x = element_text(face = "bold", size = 10, angle = 90, hjust = 1),  
    axis.text.y = element_text(face = "bold", size = 12),
    axis.title = element_text(face = "bold", size = 14),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    legend.position = "none"
  ) +
  xlab("Year") +
  ylab("Number of Outbreaks") +
  ggtitle("Year-wise Distribution of HPAI Outbreaks")

# Save  
ggsave(
  filename = file.path(output_folder, "Yearly_Outbreaks.tiff"),
  plot = p3,
  width = 10,
  height = 8,
  dpi = 600,
  device = "tiff",
  compression = "lzw"
)

# =============================================================================
# 8. Monthly Outbreak Distribution Maps
# =============================================================================

india_dist <- st_read(shapefile_path)

# Fix CRS
india_dist <- st_transform(india_dist, crs = 4326)

# Convert to sf object
df_sf <- df %>%
  filter(!is.na(Latitude), !is.na(Longitude), !is.na(Observation_date)) %>%
  st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326)


# Use existing Month column
month_levels <- 1:12
month_names <- month.abb

plots <- list()


# Monthly maps
for (i in month_levels) {
  
  month_data <- df_sf %>% filter(Month == i)
  
  p <- ggplot() +
    geom_sf(data = india_dist, fill = "skyblue", color = "black", size = 0.2) +
    geom_sf(data = month_data, color = "red", size = 1.2, alpha = 0.8) +
    theme_minimal() +
    labs(
      title = paste0(month_names[i], " Outbreaks"),
      x = "Longitude",
      y = "Latitude"
    ) +
    theme(
      axis.text = element_text(face = "bold", size = 8),
      axis.title = element_text(face = "bold", size = 10),
      panel.grid = element_blank(),
      plot.title = element_text(face = "bold", size = 12, hjust = 0.5)
    )
  
  plots[[i]] <- p
}


# Combine plots (3x4)
combined_plot <- (plots[[1]] | plots[[2]] | plots[[3]] | plots[[4]]) /
  (plots[[5]] | plots[[6]] | plots[[7]] | plots[[8]]) /
  (plots[[9]] | plots[[10]] | plots[[11]] | plots[[12]])


# Save 

ggsave(
  filename = "C:/Users/HPC/Downloads/HPAI_Project/HPAI_EDA_New/India_Monthly_Outbreaks.tiff",
  plot = combined_plot,
  width = 15,
  height = 15,
  dpi = 600,
  device = "tiff",
  compression = "lzw"
)

# =============================================================================
# 9. Seasonal Outbreak Trends Across Time Periods
# =============================================================================


# Create Period

df <- df %>%
  mutate(
    Year = year(Observation_date),
    Period = case_when(
      Year >= 2006 & Year <= 2012 ~ "First Period (2006–12)",
      Year >= 2013 & Year <= 2019 ~ "Second Period (2013–19)",
      Year >= 2020 & Year <= 2025 ~ "Third Period (2020–25)",
      TRUE ~ NA_character_
    )
  )

df$Period <- factor(
  df$Period,
  levels = c("First Period (2006–12)",
             "Second Period (2013–19)",
             "Third Period (2020–25)")
)


# Summarize
df_summary <- df %>%
  filter(!is.na(Period)) %>%
  group_by(Year, Season, Period) %>%
  summarise(Outbreaks = n(), .groups = "drop")


# Plot
p4 <- ggplot(df_summary, aes(x = factor(Year), y = Outbreaks, fill = Season)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  
  facet_grid2(
    . ~ Period,
    scales = "free_x",
    space = "free_x",
    strip = strip_themed(
      background_x = list(
        element_rect(fill = "#FFEBEE"),
        element_rect(fill = "#E3F5FD"),
        element_rect(fill = "#E9F2E9")
      )
    )
  ) +
  
  scale_fill_manual(values = c(
    "Winter"  = "red",
    "Summer"  = "blue",
    "Monsoon" = "green",
    "Autumn"  = "orange"
  )) +
  
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  
  xlab("Year") +
  ylab("Number of Outbreaks") +
  ggtitle("Season-wise HPAI Outbreaks Across Three Time Periods") +
  
  theme_minimal(base_size = 12) +
  theme(
    panel.background = element_blank(),         
    plot.background = element_rect(fill = "white", color = NA),  
    panel.grid = element_blank(),
    panel.border = element_blank(),
    panel.spacing = unit(0, "lines"),
    
    strip.background = element_blank(),          
    strip.text.x = element_text(face = "bold", size = 11),  
    
    axis.text.x = element_text(face = "bold", angle = 90, hjust = 1),
    axis.text.y = element_text(face = "bold"),
    axis.title = element_text(face = "bold"),
    
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    
    legend.title = element_blank(),
    legend.text = element_text(face = "bold")
  )

# Save

ggsave(
  filename = "C:/Users/HPC/Downloads/HPAI_Project/HPAI_EDA_New/Three_period_Outbreaks_AllData.tiff",
  plot = p4,
  width = 12,
  height = 8,
  dpi = 600,
  device = "tiff",
  compression = "lzw"
)


# =============================================================================
# 10. State-Level HPAI Outbreak Map
# =============================================================================


#  Filter HPAI serotypes
df_filtered <- df %>%
  filter(Serotype %in% c("H5N1 HPAI", "H5N8 HPAI"))


#  Standardise case first
india_grid$ST_NM <- toupper(trimws(india_grid$ST_NM))
df_filtered$States_clean <- toupper(trimws(df_filtered$States))


#  State name mapping (apply AFTER uppercase)
state_mapping <- c(
  "UTTAR PRADESH" = "UTTAR PRADESH",
  "KARNATAKA" = "KARNATAKA",
  "BIHAR" = "BIHAR",
  "MAHARASHTRA" = "MAHARASHTRA",
  "JHARKHAND" = "JHARKHAND",
  "MADHYA PRADESH" = "MADHYA PRADESH",
  "RAJASTHAN" = "RAJASTHAN",
  "KERALA" = "KERALA",
  "WEST BENGAL" = "WEST BENGAL",
  "PUNJAB" = "PUNJAB",
  "HIMACHAL PRADESH" = "HIMACHAL PRADESH",
  "HARYANA" = "HARYANA",
  "GUJARAT" = "GUJARAT",
  "TRIPURA" = "TRIPURA",
  "MANIPUR" = "MANIPUR",
  "MEGHALAYA" = "MEGHALAYA",
  "ASSAM" = "ASSAM",
  "SIKKIM" = "SIKKIM",
  "MIZORAM" = "MIZORAM",
  "TAMIL NADU" = "TAMIL NADU",
  "UTTARAKHAND" = "UTTARAKHAND",
  "ANDHRA PRADESH" = "ANDHRA PRADESH",
  "ODISHA" = "ODISHA",
  "ORISSA" = "ODISHA",
  "TELANGANA" = "TELANGANA",
  "JAMMU AND KASHMIR" = "JAMMU & KASHMIR",
  "NCT OF DELHI" = "DELHI",
  "DELHI" = "DELHI",
  "DAMAN AND DIU" = "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
  "CHATTISGARH" = "CHHATTISGARH",
  "CHHATTISGARH" = "CHHATTISGARH",
  "CHANDIGARH" = "CHANDIGARH"
)

df_filtered$States_clean <- recode(
  df_filtered$States_clean,
  !!!state_mapping
)


#  check mismatches
setdiff(unique(df_filtered$States_clean), unique(india_grid$ST_NM))


#  State-level outbreak counts
state_summary <- df_filtered %>%
  group_by(States_clean) %>%
  summarise(Total_Cases = n(), .groups = "drop")
state_summary


#  Join with shapefile
india_states_summary <- india_grid %>%
  group_by(ST_NM) %>%
  summarise() %>%
  left_join(state_summary, by = c("ST_NM" = "States_clean"))

india_states_summary$Total_Cases[is.na(india_states_summary$Total_Cases)] <- 0
india_states_summary


#  Plot map
p_map <- ggplot(india_states_summary) +
  geom_sf(aes(fill = Total_Cases), color = "black", size = 0.2) +
  
  scale_fill_gradientn(
    colours = c("#FFE5D5", "#FDAE6B", "#FC4E2A", "#B10026"),
    limits = c(0, max(india_states_summary$Total_Cases, na.rm = TRUE)),  
    oob = scales::squish,
    name = "HPAI Outbreaks"
  ) +
  
  theme_minimal(base_size = 14) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),   
    plot.background  = element_rect(fill = "white", color = NA),
    panel.grid = element_blank(),
    
    axis.title = element_blank(),
    axis.text = element_blank(),
    
    legend.title = element_text(size = 14, face = "bold"),
    legend.text  = element_text(size = 10),
    
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16)
  ) +
  
  ggtitle("State-wise Distribution of HPAI Outbreaks in India")


# Save 

output_folder <- "C:/Users/HPC/Downloads/HPAI_Project/HPAI_EDA_New"

ggsave(
  filename = file.path(output_folder, "High_Risk_States.tiff"),
  plot = p_map,
  width = 10,
  height = 8,
  dpi = 600,
  device = "tiff",
  compression = "lzw"
)

# =============================================================================
# 11. Serotype Distribution by Animal Type
# =============================================================================

table(df$Serotype, df$Animal.type)

library(dplyr)
library(ggplot2)

df_plot <- df %>%
  filter(Serotype %in% c("H5N1 HPAI", "H5N8 HPAI"),
         Animal.type %in% c("Domestic", "Wild")) %>%
  group_by(Animal.type, Serotype) %>%
  summarise(Count = n(), .groups = "drop")



p_bar <- ggplot(df_plot, aes(x = Animal.type, y = Count, fill = Serotype)) +
  geom_bar(
    stat = "identity",
    position = position_dodge(width = 0.6),
    width = 0.6
  ) +
  
  scale_fill_manual(
    values = c(
      "H5N1 HPAI" = "darkred",
      "H5N8 HPAI" = "steelblue"
    ),
    labels = c("H5N1", "H5N8"),
    name = "Serotype"
  ) +
  
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  
  theme_minimal(base_size = 14) +
  theme(
    panel.background = element_blank(),
    plot.background  = element_rect(fill = "white", color = NA),
    panel.grid = element_blank(),
    
    axis.line = element_line(color = "black", linewidth = 0.8),  
    
    axis.text = element_text(face = "bold"),
    axis.title = element_text(face = "bold"),
    
    legend.title = element_text(face = "bold"),
    legend.text = element_text(face = "bold")
  ) +
  
  xlab("Animal Type") +
  ylab("Number of Cases")



output_folder <- "C:/Users/HPC/Downloads/HPAI_Project/HPAI_EDA_New"

# save 
ggsave(
  filename = file.path(output_folder, "Serotype_AnimalType.tiff"),
  plot = p_bar,
  width = 8,
  height = 6,
  dpi = 600,
  device = "tiff",
  compression = "lzw"
)
