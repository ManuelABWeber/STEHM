# cd "C:\0_Documents\10_ETH\Thesis\Python"
# env\Scripts\activate

'''
from ultralytics import YOLO
import os
import pandas as pd
from PIL import Image, ExifTags
from datetime import datetime

def check_and_remove_corrupted_files(directory_path):
    count = 0
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:  # Open and check the image
                        img.verify()  # Verify the image is intact
                except (IOError, SyntaxError) as e:
                    print(f"Corrupted file detected and removed: {file_path}")
                    os.remove(file_path)  # Remove the corrupted file
                    count += 1
    print(f"Total corrupted files removed: {count}")

def get_image_metadata(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == "DateTimeOriginal":
                    return value
    except Exception as e:
        print(f"Error extracting metadata: {e}")
    return None

model = YOLO("yoloweights.pt")

image_path = "C:/0_Documents/10_ETH/Thesis/WPS images"
check_and_remove_corrupted_files(image_path)  # Remove corrupted files before analysis

results = model.predict(source=image_path + "/*",  # Specify the correct path for the images, /*/* for 2 folder system
                        stream=True,
                        save=True,
                        conf = 0.75)  # Do not save the images again

# Get list of species names from the model
species_names = list(model.names.values())

# Initialize DataFrame to store results
columns = ['Name', 'Date_Time'] + species_names
df = pd.DataFrame(columns=columns)

for r in results:
    date_time = get_image_metadata(r.path)
    if date_time is None:
        print(f"Metadata not found for image: {r.path}")
        continue

    # Initialize species detection dictionary
    species_detections = {name: 0 for name in species_names}
    conf = r.boxes.conf.numpy()
    for idx, t in enumerate(r.boxes.cls.numpy()):
        species = model.names[int(t)]
        if conf[idx] > species_detections[species]:
            species_detections[species] = conf[idx]

    # Create row for the current image
    row = [r.path, date_time] + [species_detections[species] for species in species_names]
    df.loc[len(df)] = row

csv_file = "C:/0_Documents/Onguma_W33.csv"
df.to_csv(csv_file, index=False)
print(f"CSV file saved: {csv_file}")


### Validation on test split YOLOv8
from ultralytics import YOLO

# Load a model
model = YOLO("yoloweights.pt")  # load a custom model

# Validate the model
metrics = model.val(data = 'valid.yaml', plots = True, conf=0.75 )  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
'''

### Validation on test split YOLOv10
from ultralytics import YOLOv10

model = YOLOv10('YOLOv10weights/epoch40.pt')

#model = YOLOv10.from_pretrained('YOLOv10weights/yolov10l')

metrics = model.val(data='valid.yaml', batch=16, conf=0.75)
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category