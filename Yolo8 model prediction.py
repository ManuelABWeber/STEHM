# cd "C:\0_Documents\10_ETH\Thesis\Python"
# env\Scripts\activate


from ultralytics import YOLO
import os
import pandas as pd
from PIL import Image, ExifTags
from datetime import datetime

def check_and_remove_corrupted_files(directory_path):
    """Checks for corrupted image files in the specified directory and removes them."""
    count = 0
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:  # Open and check the image
                        img.verify()  # Verify the image is intact
                except (IOError, SyntaxError) as e:
                    print(f"Corrupted file detected and removed: {file_path}")
                    os.remove(file_path)  # Remove the corrupted file
                    count += 1
    print(f"Total corrupted files removed: {count}")

# Function to extract image metadata (date and time)
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

model = YOLO("prelweights_1704.pt")

image_path = "D:/Manuel Onguma/2022/sorted/ONG"
check_and_remove_corrupted_files(image_path)  # Remove corrupted files before analysis

results = model.predict(source=image_path + "/*",  # Specify the correct path for the images, /*/* for 2 folder system
                        stream=True,
                        save=False)  # Do not save the images again

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

csv_file = "C:/0_Documents/Ongava_2022ONG.csv"
df.to_csv(csv_file, index=False)
print(f"CSV file saved: {csv_file}")



############################
'''

from ultralytics import YOLO
import os
import pandas as pd
from PIL import Image, ExifTags
from datetime import datetime

# Initialize YOLO model
model = YOLO("prelweights_1704.pt")

def check_corrupted_files(directory_path):
    valid_images = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)  # Attempt to open the image
                    img.verify()  # Verify the image contents
                    valid_images.append(file_path)  # Append the valid image path to the list
                except (IOError, SyntaxError) as e:
                    print(f"Corrupted file: {file_path}")
    return valid_images

def get_image_metadata(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == "DateTimeOriginal":
                    return value
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
    return None

image_path = "D:/Manuel Onguma/2022"
valid_image_paths = check_corrupted_files(image_path)  # Get all valid image paths

results = model.predict(source=valid_image_paths,  # Use list of valid paths
                        stream=True,
                        save=False)

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
    species_detections = {species: 0 for species in species_names}
    conf = r.boxes.conf.numpy()
    for idx, t in enumerate(r.boxes.cls.numpy()):
        t = int(t)  # Convert tensor to integer
        if conf[idx] > species_detections[species_names[t]]:
            species_detections[species_names[t]] = conf[idx]

    # Create row for the current image
    row = [r.path, date_time] + list(species_detections.values())

    # Append row to DataFrame
    df.loc[len(df)] = row

# Write DataFrame to CSV file
csv_file = "C:/0_Documents/Ongava_2022ALL.csv"
df.to_csv(csv_file, index=False)

print(f"CSV file saved: {csv_file}")



### Validation on test split
from ultralytics import YOLO

# Load a model
model = YOLO("prelweights_1704.pt")  # load a custom model

# Validate the model
metrics = model.val(data = 'testdata.yaml', plots = True, )  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
'''