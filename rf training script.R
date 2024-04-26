### ------------------------------------------------------------------------------------------------------------------
### Task 1: Growth form fractions from aerial photograph ------------------------------------
### ------------------------------------------------------------------------------------------------------------------

rm(list = ls())

# Load required libraries
library(terra)
library(ranger)

# Working directory
setwd("C:/0_Documents/10_ETH/Thesis/GIS/Growth form model")

# Reading in the training data (manully obtained point samples from QGIS)
plots1 <- read.csv("trainingset_tile1.csv")
orthophoto1 <- rast("maptile1_georeferenced.tif") # aerial image

plots2 <- read.csv("trainingset_tile2.csv")
orthophoto2 <- rast("maptile2_georeferenced.tif") # aerial image

plots3 <- read.csv("trainingset_tile3.csv")
orthophoto3 <- rast("maptile3_georeferenced.tif") # aerial image


### The following section has been run and all changes have been saved.
### Add 3 additional predictors
orthophoto1 <- orthophoto1[[c("R","G","B")]] # extract rgb channels
orthophoto2 <- orthophoto2[[c("R","G","B")]] # extract rgb channels
orthophoto3 <- orthophoto3[[c("R","G","B")]] # extract rgb channels
orthophotovalues1 <- values(orthophoto1)
orthophotovalues2 <- values(orthophoto2)
orthophotovalues3 <- values(orthophoto3)
orthophotovalues1 <- as.data.frame(orthophotovalues1)
orthophotovalues2 <- as.data.frame(orthophotovalues2)
orthophotovalues3 <- as.data.frame(orthophotovalues3)
#names(orthophotovalues1) <- c("R","G","B")
#names(orthophotovalues2) <- c("R","G","B")
#names(orthophotovalues3) <- c("R","G","B")
# 1. Add the visible vegetation index as predictor (Higginbottom et al. 2018)
# R0, G0, B0 are constants used to reference green color (Joseph and Devadas 2015)
calculate_VVI <- function(df, R0 = 30, G0 = 50, B0 = 1) {
  VVI <- ((1 - df$R - R0) * (1 - df$G - G0) * (1 - df$B - B0)) / ((df$R + R0) * (df$G + G0) * (df$B + B0))
  return(VVI)
}

VVI1 <- calculate_VVI(orthophotovalues1)
VVI2 <- calculate_VVI(orthophotovalues2)
VVI3 <- calculate_VVI(orthophotovalues3)

# 2. Add the values of the two first principle components of a PCA run on the images (Higginbottom et al. 2018)
# Perform PCA on the matrix
perform_PCA <- function(matrix_data) {
  # Center the data
  centered_data <- scale(matrix_data, center = TRUE, scale = FALSE)

  # Compute covariance matrix
  cov_matrix <- cov(centered_data)

  # Perform PCA
  pca_result <- prcomp(centered_data, scale. = FALSE)

  return(pca_result)
}

# Call the function to perform PCA
pca_result1 <- perform_PCA(matrix_data = orthophotovalues1)
pca_result2 <- perform_PCA(matrix_data = orthophotovalues2)
pca_result3 <- perform_PCA(matrix_data = orthophotovalues3)
summary(pca_result1)
summary(pca_result2)
summary(pca_result3)
# 
# # Extract the loadings (eigenvectors) of the first two principal components
# loadings_first_two_pc1 <- pca_result1$rotation[, 1:2]
# loadings_first_two_pc2 <- pca_result2$rotation[, 1:2]
# loadings_first_two_pc3 <- pca_result3$rotation[, 1:2]
# 
# # Project the original dataframe onto the first two principal components
# pcdf1 <- as.data.frame(predict(object = pca_result1, orthophotovalues1)[, 1:2])
# pcdf2 <- as.data.frame(predict(object = pca_result2, orthophotovalues2)[, 1:2])
# pcdf3 <- as.data.frame(predict(object = pca_result3, orthophotovalues3)[, 1:2])
# orthophotovalues1$PC1 <- pcdf1$PC1
# orthophotovalues2$PC1 <- pcdf2$PC1
# orthophotovalues3$PC1 <- pcdf3$PC1
# orthophotovalues1$PC2 <- pcdf1$PC2
# orthophotovalues2$PC2 <- pcdf2$PC2
# orthophotovalues3$PC2 <- pcdf3$PC2
# orthophotovalues1$VVI <- VVI1
# orthophotovalues2$VVI <- VVI2
# orthophotovalues3$VVI <- VVI3
# 
# # Add the predictors to the 3 rasters
# # Add three bands
# orthophoto1b <- orthophoto1
# orthophoto2b <- orthophoto2
# orthophoto3b <- orthophoto3
# values(orthophoto1) <- orthophotovalues1[,1:3]
# values(orthophoto2) <- orthophotovalues2[,1:3]
# values(orthophoto3) <- orthophotovalues3[,1:3]
# values(orthophoto1b) <- orthophotovalues1[,4:6]
# values(orthophoto2b) <- orthophotovalues2[,4:6]
# values(orthophoto3b) <- orthophotovalues3[,4:6]
# orthophoto1 <- c(orthophoto1,orthophoto1b)
# orthophoto2 <- c(orthophoto2,orthophoto2b)
# orthophoto3 <- c(orthophoto3,orthophoto3b)
# names(orthophoto1) <- c("R","G","B","PC1","PC2","VVI")
# names(orthophoto2) <- c("R","G","B","PC1","PC2","VVI")
# names(orthophoto3) <- c("R","G","B","PC1","PC2","VVI")
# 
# writeRaster(orthophoto1, "GIS/maptile1_georeferenced.tif", overwrite = T)
# writeRaster(orthophoto2, "GIS/maptile2_georeferenced.tif", overwrite = T)
# writeRaster(orthophoto3, "GIS/maptile3_georeferenced.tif", overwrite = T)

# Nomenclature
# 1: Woody
# 2: Herbaceous
# 3: Bare

# Step 1: Prepare the data

# Growth form types
habitat_points1 <- plots1$class
habitat_points2 <- plots2$class
habitat_points3 <- plots3$class

# Extract the pixel values of the sentinel bands for the chosen locations
chosen_locations1 <- data.frame(x = plots1$Longitude, y = plots1$Latitude)
chosen_locations2 <- data.frame(x = plots2$Longitude, y = plots2$Latitude)
chosen_locations3 <- data.frame(x = plots3$Longitude, y = plots3$Latitude)
data1 <- extract(orthophoto1, chosen_locations1, ID = F)
data2 <- extract(orthophoto2, chosen_locations2, ID = F)
data3 <- extract(orthophoto3, chosen_locations3, ID = F)

# Combine the growth form classes and the channels of the orthophoto into one dataframe
df1 <- data.frame(habitat_points1, data1)
df2 <- data.frame(habitat_points2, data2)
df3 <- data.frame(habitat_points3, data3)
names(df1) <- c("habitat_points", "R", "G", "B", "PC1", "PC2", "VVI")
names(df2) <- c("habitat_points", "R", "G", "B", "PC1", "PC2", "VVI")
names(df3) <- c("habitat_points", "R", "G", "B", "PC1", "PC2", "VVI")
df <- rbind(df1,df2,df3)
df$habitat_points <- as.factor(df$habitat_points)

# Step 2: Train the random forest model

# Split the data into training and testing sets
set.seed(123)  # For reproducibility
train_indices <- sample(1:nrow(df), nrow(df) * 0.8)  # 80% for training
train_data <- df[train_indices, ]
train_data <- na.omit(train_data)
test_data <- df[-train_indices, ]

# Train the random forest model
rf_model <- ranger(habitat_points ~ ., data = train_data, num.trees = 500)

# Validation
# Extract the test data and actual tree cover values
test_data <- na.omit(test_data)
test_input <- test_data[, -1]  # Remove the habitat_points column
actual_habitat_points <- test_data$habitat_points

# Predict tree cover for the test set using the trained model
predicted_habitat_points <- predict(rf_model, data = test_input)$predictions

# Evaluate the performance of the model
accuracy <- mean(predicted_habitat_points == actual_habitat_points)
print(paste("Accuracy:", accuracy)) # 94% accuracy

# Predictions
orthophoto <- orthophoto3 # run the following section for all 3 images

# Prediction
# Define the size of the chunks in degrees based on memory constraints and raster resolution
chunk_size_x <- 0.01
chunk_size_y <- 0.01

# Calculate the number of chunks in each dimension
num_chunks_x <- ceiling((xmax(orthophoto)-xmin(orthophoto)) / chunk_size_x)
num_chunks_y <- ceiling((ymax(orthophoto)-ymin(orthophoto)) / chunk_size_y)

# Initialize an empty list to store the predicted chunks
predicted_chunks <- list()

# Define helper function
pfun <- \(...) {
  predict(...)$predictions
}

# Loop through each chunk
for (i in 1:num_chunks_x) {
  for (j in 1:num_chunks_y) {
    # Define the extent of the current chunk in geographic units
    xmin <- xmin(orthophoto) + (i - 1) * chunk_size_x
    xmax <- min(xmax(orthophoto), xmin + chunk_size_x)
    ymin <- ymin(orthophoto) + (j - 1) * chunk_size_y
    ymax <- min(ymax(orthophoto), ymin + chunk_size_y)
    
    # Create an extent object for the chunk
    chunk_extent <- ext(c(xmin, xmax, ymin, ymax))
    
    # Crop the raster to extract the chunk
    chunk <- crop(orthophoto, chunk_extent)
    
    # Make predictions on the chunk
    predicted_chunk <- predict(chunk,rf_model, fun = pfun, na.rm = T)
    
    # Store the predicted chunk
    predicted_chunks[[length(predicted_chunks) + 1]] <- predicted_chunk
  }
}

# Create a SpatRasterCollection
s <- sprc(predicted_chunks)

# Merge the rasters
m <- merge(s)

# Store the output
writeRaster(m,  "predicted_growthforms_tile3.tif", overwrite = T)

### ------------------------------------------------------------------------------------------------------------------
### Task 2: Aggregating growth form fractions to sentinel2 raster template and training data extraction --------------
### ------------------------------------------------------------------------------------------------------------------

rm(list = ls())
setwd("C:/0_Documents/10_ETH/Thesis/GIS/Growth form model")
library(terra)
#library(sf)
#onguma <- read_sf("C:/0_Documents/10_ETH/Thesis/GIS/Onguma_sf/Onguma_EPSG_4326.shp")[1]

## Reading in the data
raster_cat1 <- rast("predicted_growthforms_tile1.tif")
raster_cat2 <- rast("predicted_growthforms_tile2.tif")
raster_cat3 <- rast("predicted_growthforms_tile3.tif")

summary(raster_cat3)

sentinel1 <- rast("Sentinel-2_Image_2302_med.tif") # september 2023 (when aerial images were taken)
sentinel2 <- rast("Sentinel-2_Image_2502_med_2022.tif") # september 2022 (one year before aerial images were taken)
sentinel3 <- rast("Sentinel-2_Image_2502_med_rainy.tif") # january 2023 (rainy season before images were taken)


dl_training_generation <- function(raster_cat, sentinelraw, outputname){
  
  ### Sentinel2 imagery normalization from Brown et al. 2022 - the numbers are very small, maybe check
  # Define normalization percentiles as a matrix
  NORM_PERCENTILES <- matrix(c(
    1.7417268007636313, 2.023298706048351,
    1.7261204997060209, 2.038905204308012,
    1.6798346251414997, 2.179592821212937,
    1.7734969472909623, 2.2890068333026603,
    2.289154079164943, 2.6171674549378166,
    2.382939712192371, 2.773418590375327,
    2.3828939530384052, 2.7578332604178284,
    2.1952484264967844, 2.789092484314204,
    1.554812948247501, 2.4140534947492487), ncol = 2, byrow = TRUE)
  
  # Normalize and apply sigmoid transfer function
  normalize_and_transfer <- function(x) {
    x <- log(x * 0.005 + 1)
    x <- (x - NORM_PERCENTILES[band_index,1]) / NORM_PERCENTILES[band_index,2]
    x <- exp(x * 5 - 1)
    x <- x / (x + 1)
  }
  
  sentinel <- crop(sentinelraw, raster_cat)
  #plot(sentinel$B2)
  
  # Convert SpatRaster to an array
  image_data_array <- as.array(sentinel)
  
  # Apply the function to the array
  for (band_index in 1:dim(image_data_array)[3]) {
    image_data_array[,,band_index] <- normalize_and_transfer(image_data_array[,,band_index])
  }
  
  # Assign the modified array back to the SpatRaster object
  values(sentinel) <- image_data_array
  
  ### Aerial image preprocessing
  #1. transform input to 3 binary bands
  band1 <- raster_cat # woody
  band1[band1 == 1] <- 1
  band1[band1 == 2] <- 0
  band1[band1 == 3] <- 0
  band2 <- raster_cat # herbaceous
  band2[band2 == 1] <- 0
  band2[band2 == 2] <- 1
  band2[band2 == 3] <- 0
  band3 <- raster_cat # bare
  band3[band3 == 1] <- 0
  band3[band3 == 2] <- 0
  band3[band3 == 3] <- 1
  
  #2. reproject it using nearest neighbor on multiple of resolution of sentinel band
  new_raster <- rast(nrow = nrow(sentinel)*10, ncol = ncol(sentinel)*10, 
                     xmin = xmin(sentinel), ymin = ymin(sentinel), 
                     xmax = xmax(sentinel), ymax = ymax(sentinel), 
                     crs = crs(sentinel))
  
  band1 <- resample(band1, new_raster, method = "near")
  band2 <- resample(band2, new_raster, method = "near")
  band3 <- resample(band3, new_raster, method = "near")
  
  binary_raster <- rast(list(band1,band2,band3))
  names(binary_raster) <- c("Woody", "Herbaceous", "Bare")
  #plot(binary_raster)
  
  #3. aggregate the raster up to sentinel resolution and normalize to 0-1
  growthforms <- aggregate(binary_raster, fact = 10, fun = sum, na.rm=T)/100
  #plot(growthforms)
  output <- rast(list(sentinel,growthforms))
  summary(output)
  
  #4. create training data for deep learning model (2500 points randomly sampled, 2500 points where the growth forms are <0.1 or >0.9)
  randomsample <- spatSample(output, 2500, method = "random", replace = F, na.rm = T, as.df = T)
  filtered_points <- output
  
  # Set threshold values
  threshold <- 0.1
  
  # Create a condition raster
  condition <- filtered_points$Bare < threshold | filtered_points$Bare > (1 - threshold) |
    filtered_points$Herbaceous < threshold | filtered_points$Herbaceous > (1 - threshold) |
    filtered_points$Woody < threshold | filtered_points$Woody > (1 - threshold)
  
  # Filter the spatraster based on the condition raster
  filtered_points <- mask(filtered_points, condition)
  
  # Sample points from the SpatialPointsDataFrame
  stratsample <- spatSample(filtered_points, 2500, method = "random", replace = FALSE, na.rm = TRUE, as.df = TRUE)
  

 randomsample <- rbind(randomsample, stratsample)
  x <- randomsample[,1:9]
  y <- randomsample[,10:12]
  
  write.csv(x, paste0("x_", outputname, ".csv"))
  write.csv(y, paste0("y_", outputname, ".csv")) 
}

dl_training_generation(raster_cat1, sentinel1, "tile1_sept2023")
dl_training_generation(raster_cat1, sentinel2, "tile1_sept2022")
dl_training_generation(raster_cat1, sentinel3, "tile1_jan2023")
dl_training_generation(raster_cat2, sentinel1, "tile2_sept2023")
dl_training_generation(raster_cat2, sentinel2, "tile2_sept2022")
dl_training_generation(raster_cat2, sentinel3, "tile2_jan2023")
dl_training_generation(raster_cat3, sentinel1, "tile3_sept2023")
dl_training_generation(raster_cat3, sentinel2, "tile3_sept2022")
dl_training_generation(raster_cat3, sentinel3, "tile3_jan2023")

x <- rbind(read.csv("x_tile1_sept2023.csv"),
           read.csv("x_tile1_sept2022.csv"),
           read.csv("x_tile1_jan2023.csv"),
           read.csv("x_tile2_sept2023.csv"),
           read.csv("x_tile2_sept2022.csv"),
           read.csv("x_tile2_jan2023.csv"),
           read.csv("x_tile3_sept2023.csv"),
           read.csv("x_tile3_sept2022.csv"),
           read.csv("x_tile3_jan2023.csv"))[,-1]
write.csv(x, "finaltraining_x.csv")

y <- rbind(read.csv("y_tile1_sept2023.csv"),
           read.csv("y_tile1_sept2022.csv"),
           read.csv("y_tile1_jan2023.csv"),
           read.csv("y_tile2_sept2023.csv"),
           read.csv("y_tile2_sept2022.csv"),
           read.csv("y_tile2_jan2023.csv"),
           read.csv("y_tile3_sept2023.csv"),
           read.csv("y_tile3_sept2022.csv"),
           read.csv("y_tile3_jan2023.csv"))[,-1]
write.csv(y, "finaltraining_y.csv")

summary(y)

hist(y$Woody)
hist(y$Herbaceous)
hist(y$Bare)
