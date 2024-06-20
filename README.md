# Spatially and Temporally Explicit Herbivory Monitoring in Water-limited Savannas
Effective management of protected areas is critical to mitigating the global biodiversity crisis. In water-limited trophic savannas, altered herbivory regimes lead to ecosystem degradation. Here, we present an innovative tool that integrates a YOLO detection model applied to waterpoint-based camera trap images to detect herbivores with a deep-learning model to estimate vegetation category fractions from Sentinel-2 imagery. This tool allows for monitoring of herbivore densities and biomass availability across finer spatial and temporal scales than previously possible. By tying data collection to surface water, a primary determinant of large herbivore distribution in semi-arid environments, our framework enables adaptive herbivory management and provides an approach to disentangle the ecological drivers influencing plant-herbivore interactions. This refined monitoring capability can contribute to enhance conservation strategies and promote the restoration of savannas.

This repository contains the source code of the app, as well as training and prediction scripts for the models.

![Methods PP figure 2006](https://github.com/Manuel-Weber-ETH/cameratraps/assets/118481837/4965cdf4-8590-4628-883b-5f28ad5c8910)
Figure 1: Methodology

## Files:
Vegetation category model:
- Random forest model training script: "rf training script.R"
- Deep learning model training script: "DL vegetation model training.py"

YOLO model:
- Training script: "Python Script - Yolo Training.ipynb"
- Prediction script: "Yolo model prediction.py"

Time window analysis:
- Script: "Time window analysis.R"

Software:
- Script: "app.py"

![conceptual model msc 2006](https://github.com/Manuel-Weber-ETH/cameratraps/assets/118481837/d9377621-45e2-476b-90db-b90b8d7b7600)
Figure 2: Conceptual model
