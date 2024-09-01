########################################################################################################################################################################################
### Spatially and Temporally Explicit Herbivory Monitoring in Water-limited Savannas
### YOLO model training
### Manuel Weber
### https://github.com/ManuelABWeber/STEHM.git
########################################################################################################################################################################################

#!pip install ultralytics
#!nvidia-smi
from ultralytics import YOLO
model = YOLO('yolov10l.pt')

results = model.train(data='/content/drive/MyDrive/data.yaml',
                      epochs=100,
                      patience=15,
                      imgsz=640,
                      batch=16,
                      name="yolo10_STEHM",
                      plots=True,
                      save_period=10,  # Save a checkpoint every 10 epochs.
                      fliplr=0.5,  # Horizontal flip probability
                      scale=0.3,  # Random scaling
                      perspective=0.0003,  # Slight perspective change
                      translate=0.2,  # Random translation
                      hsv_h=0.015,  # Hue shift
                      hsv_s=0.7,  # Saturation fluctuation
                      hsv_v=0.4,  # Value fluctuation
                  )