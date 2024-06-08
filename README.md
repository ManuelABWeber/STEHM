# Spatially and Temporally Explicit Herbivory Monitoring in Water-limited Savannas
Effective management of protected areas is critical to mitigating the global biodiversity crisis. In water-limited trophic savannas, altered herbivory regimes lead to ecosystem degradation. Here, we present an innovative tool that integrates a YOLO detection model applied to waterpoint-based camera trap images to detect herbivores with a deep-learning model to estimate vegetation category fractions from Sentinel-2 imagery. This tool allows for monitoring of herbivore densities and biomass availability across finer spatial and temporal scales than previously possible. By tying data collection to surface water, a primary determinant of large herbivore distribution in semi-arid environments, our framework enables adaptive herbivory management and provides an approach to disentangle the ecological drivers influencing plant-herbivore interactions. This refined monitoring capability can contribute to enhance conservation strategies and promote the restoration of savannas.

This repository contains the source code of the app, as well as training and prediction scripts for the various models.

![Methods PP figure](https://github.com/Manuel-Weber-ETH/cameratraps/assets/118481837/778b713a-06b3-48d8-b2e8-914b228b108d)
Figure 1: Methodology

## Files:
Vegetation category model:
- Random forest model training script: "rf training script.R"
- Deep learning model training script: "DL vegetation model training.py"

YOLOv8 model:
- Training script: "Python Script - Yolo8 Training.ipynb"
- Prediction script: "Yolo8 model prediction.py"

Time window analysis:
- Script: "Time window analysis.R"

Software:
- Script: "app.py"

![conceptual model msc 2604](https://github.com/Manuel-Weber-ETH/cameratraps/assets/118481837/98a97cdc-4330-45bf-8aec-6de10703476b)
Figure 2: Conceptual model
