# Spatially and Temporally Explicit Herbivory Monitoring in Water-limited Savannas
Effective management of protected areas is critical to mitigating the global biodiversity crisis. In water-limited trophic savannas, altered herbivory regimes lead to ecosystem degradation. Here, we present an innovative tool that integrates a YOLO detection model applied to waterpoint-based camera trap images to detect herbivores with a deep-learning model to estimate vegetation category fractions from Sentinel-2 imagery. This tool allows for monitoring of herbivore densities and biomass availability across finer spatial and temporal scales than previously possible. By tying data collection to surface water, a primary determinant of large herbivore distribution in semi-arid environments, our framework enables adaptive herbivory management and provides an approach to disentangle the ecological drivers influencing plant-herbivore interactions. This refined monitoring capability can contribute to enhance conservation strategies and promote the restoration of savannas.

This repository contains the source code of the two desktop applications, training scripts for the models and the raw results.

![figure1](https://github.com/user-attachments/assets/c4be0c13-e30d-4c20-8982-d9babc0945da)
Figure 1: Methodology

## Files:
Vegetation category model:
- Random forest model training script: "Vegetation category models.R"
- Deep learning model training script: "vegetation model training.py"

YOLO model:
- Training script: "YOLO model training.py"

Time window analysis:
- Script: "Time window analysis.R"

Desktop applications:
- STEHM v1.0: "STEHM_v1.0.py"
- STEHM v1.0 camera trap image classifier: "STEHM_v1.0_classifier.py"

Results:
- Results Case Study Site 1: "results case study site 1.csv"
- Results Case Study Site 2: "results case study site 2.csv"

![figure5](https://github.com/user-attachments/assets/2213caa7-d3f6-4fd5-9c2a-175cafb2c3cc)
Figure 2: Conceptual model
