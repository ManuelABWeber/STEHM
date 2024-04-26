# A Framework for Herbivory Management in Water-limited Savannas
Sound protected area management is essential to address the global biodiversity crisis. Many water-limited protected areas in trophic savannas degrade due to altered herbivory regimes driven by fragmentation and artificial water supply. We introduce a tool to contextualize game densities and biomass availability in time and space using a YOLOv8 detection model applied to waterhole-based camera traps, and a deep-learning model to estimate vegetation category fractions from Sentinel-2 imagery. Since surface water availability is the primary factor that dictates the distribution of large herbivores in semi-arid environments, this creates a framework for adaptive management of herbivory, and a setting to experimentally disentangle the drivers that affect plant-herbivore interactions by refining the spatial and temporal resolutions of herbivory monitoring.

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
