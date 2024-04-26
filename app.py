import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import date, timedelta, datetime
import os
import sys
import numpy as np
import rasterio
from tensorflow.keras.models import load_model
import time
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ExifTags, ImageTk
from rasterio.mask import geometry_mask
from rasterio.windows import Window
import shapely
from scipy.spatial import Voronoi
from ultralytics import YOLO
import threading
from tkinter import ttk

script_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

classification_start_time = None

def browse_file(entry_field):
    filepath = filedialog.askopenfilename()
    entry_field.delete(0, tk.END)
    entry_field.insert(0, filepath)

# Function to read raster data and convert to array
def raster_to_array(raster_path):
    with rasterio.open(raster_path) as src:
        raster_array = src.read()
        raster_array = np.moveaxis(raster_array, 0, -1)  # Move the band axis to the last dimension
    return raster_array

# Function to apply the model to a raster array
def apply_model_to_raster(gf_model, raster_array):
    height, width, bands = raster_array.shape
    
    # Reshape raster array to a 2D array (bands x pixels) for vectorized operations
    flattened_raster = raster_array.reshape(-1, bands)

    # Predict using the model for all pixels at once
    predictions = gf_model.predict(flattened_raster)

    # Reshape predictions to the original raster shape
    output = predictions.reshape(height, width, 3)

    return output

def run_growth_form_model():
    # Start the classification in a separate thread
    threading.Thread(target=classify_raster).start()

## Main function section 1
def classify_raster(input_raster_path, progress_label):
    # Create output folder with today's date
    output_folder_name = f"run_{date.today()}"
    os.makedirs(output_folder_name, exist_ok=True)
    
    # Define input variables (will be replaced with values from GUI)
    input_raster_path = input_raster_entry.get()

    # Update progress label
    progress_label.config(text="Loading model...")

    # Construct the path to the model weights file
    model_path = os.path.join(script_dir, "growthformDLmodel")

    # Load the model using the constructed path
    gf_model = load_model(model_path)
    
    # Convert raster to array
    raster_array = raster_to_array(input_raster_path)
    
    # Update progress label
    progress_label.config(text="Raster normalization...")
    
    # Normalize the array
    NORM_PERCENTILES = np.array([
        [1.7417268007636313, 2.023298706048351],
        [1.7261204997060209, 2.038905204308012],
        [1.6798346251414997, 2.179592821212937],
        [1.7734969472909623, 2.2890068333026603],
        [2.289154079164943, 2.6171674549378166],
        [2.382939712192371, 2.773418590375327],
        [2.3828939530384052, 2.7578332604178284],
        [2.1952484264967844, 2.789092484314204],
        [1.554812948247501, 2.4140534947492487]])
    
    # Normalization + sigmoid transfer
    for band_index in range(raster_array.shape[2]):
        raster_array[:,:,band_index] = np.log(raster_array[:,:,band_index] * 0.005 + 1)
        raster_array[:,:,band_index] = (raster_array[:,:,band_index] - NORM_PERCENTILES[band_index, 0]) / NORM_PERCENTILES[band_index, 1]
        raster_array[:,:,band_index] = np.exp(raster_array[:,:,band_index] * 5 - 1)
        raster_array[:,:,band_index] = raster_array[:,:,band_index] / (raster_array[:,:,band_index] + 1)

    progress_label.config(text="Applying model... Please allow up to 1 min per 1000 ha.")

    # Apply the model to the raster array
    output_raster_array = apply_model_to_raster(gf_model, raster_array)

    progress_label.config(text="Success. Saving output raster...")
    
    # Write the output raster to a new GeoTIFF file
    output_raster_path = f"{output_folder_name}/{output_folder_name}_growth_forms_raster.tif"
    
    # Get the profile from the input raster
    with rasterio.open(input_raster_path) as src:
        profile = src.profile
    
    # Update the profile to match the shape and data type of the output raster
    profile.update(
        count=3,  # Update the number of bands to 3
        dtype=output_raster_array.dtype
    )
    
    # Write the output raster with the updated profile
    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        # Write each band separately
        for band_index in range(output_raster_array.shape[2]):
            dst.write(output_raster_array[:, :, band_index], band_index + 1)
    
    # Update progress label
    progress_label.config(text=f"Growth form raster saved at: {output_raster_path}")
    
    # Show success message
    messagebox.showinfo("Success", "Growth form raster saved successfully.")


voronoi_function_running = False

## Main function section 2
def compute_growth_form_voronoi_polygons(output_raster_path, polygon_shp, points_csv, output_folder_name):
    # Update progress label
    progress_label2.config(text="Loading growth form raster...")

    # Load classified raster
    with rasterio.open(output_raster_path) as src:
        classified_raster = src.read()
        classified_raster = np.moveaxis(classified_raster, 0, -1)  # Move the band axis to the last dimension
        transform = src.transform
        profile = src.profile

    # Load polygon from shapefile
    polygon_gdf = gpd.read_file(polygon_shp)
    polygon = polygon_gdf.geometry.iloc[0]

    # Load points from CSV
    points_df = gpd.read_file(points_csv)
    coords = np.array(points_df[['Longitude', 'Latitude']])

    # Update progress label
    progress_label2.config(text="Performing Voronoi tessellation...")

    # Create boundary polygon
    bound = polygon.buffer(100).envelope.boundary

    # Create many points along the rectangle boundary. I create one every 100 m.
    boundarypoints = [bound.interpolate(distance=d) for d in range(0, np.ceil(bound.length).astype(int), 100)]
    boundarycoords = np.array([[p.x, p.y] for p in boundarypoints])

    # Create an array of all points on the boundary and inside the polygon
    all_coords = np.concatenate((boundarycoords, coords))

    # Compute Voronoi diagram
    vor = Voronoi(points=all_coords)

    # Create Voronoi lines
    lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]

    # Polygonize lines to create Voronoi polygons
    polys = shapely.ops.polygonize(lines)

    # Create GeoDataFrame from Voronoi polygons
    voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys), crs=polygon_gdf.crs)
    # Clip Voronoi polygons by the input shapefile
    voronois = gpd.clip(voronois, polygon_gdf)

    # Update progress label
    progress_label2.config(text="Computing growth form fractions for Voronoi polygons...")

    # Overlay Voronoi polygons with input polygon to get intersection
    result = gpd.overlay(df1=voronois, df2=polygon_gdf, how="intersection")

    # Initialize lists to store mean growth form fractions for each band
    mean_band1_fractions = []
    mean_band2_fractions = []
    mean_band3_fractions = []
    area_hectares = []

    # Convert the GeoDataFrame to EPSG:3857
    # Note that I am not using a UTM crs, in a future version we may need to add the option to choose the crs since the area calculations will be different
    voronois_3857 = voronois.to_crs(epsg=3857)
    # Save Voronoi polygons as a shapefile
    voronois_3857.to_file(f"{output_folder_name}/{output_folder_name}_voronoi_polygons.shp")

    # Iterate over each Voronoi polygon
    for index, row in result.iterrows():
        polygon_3857 = voronois_3857.iloc[index]['geometry']
        # Compute the area of the polygon in hectares
        area_hectares.append(polygon_3857.area / 10000)
        polygon = row['geometry']
        # Your existing code for extracting pixel values and calculating mean growth form fractions
        mask = geometry_mask([polygon], out_shape=classified_raster.shape[:2], transform=transform, invert=True)
        masked_values = classified_raster[mask]
        mean_band1 = np.mean(masked_values[:, 0])
        mean_band2 = np.mean(masked_values[:, 1])
        mean_band3 = np.mean(masked_values[:, 2])
        mean_band1_fractions.append(mean_band1)
        mean_band2_fractions.append(mean_band2)
        mean_band3_fractions.append(mean_band3)

    # Add mean growth form fractions to the GeoDataFrame
    result['Site'] = points_df['Site']
    result['mean_woody_fraction'] = mean_band1_fractions
    result['mean_herbaceous_fraction'] = mean_band2_fractions
    result['mean_bare_fraction'] = mean_band3_fractions
    result.drop(columns=['id', 'geometry'], inplace=True)
    # Add the area values as a new column in the DataFrame
    result['area_hectares'] = area_hectares

    # Save results to a CSV file
    output_csv = f"{output_folder_name}/{output_folder_name}_growth_forms.csv"
    result.to_csv(output_csv)

    # Update progress label
    progress_label2.config(text=f"Mean growth form fractions saved to: {output_csv}")
    
    # Show success message
    messagebox.showinfo("Success", "Growth form fractions saved successfully.")


def run_voronoi_button_callback():
    # Define input variables (will be replaced with values from GUI)
    output_raster_path = f"run_{date.today()}/run_{date.today()}_growth_forms_raster.tif"
    polygon_shp = polygon_shp_entry.get()
    points_csv = points_csv_entry.get()
    output_folder_name = f"run_{date.today()}"
    
    # Check if classified raster exists
    if os.path.exists(output_raster_path):
        # Run the function
        compute_growth_form_voronoi_polygons(output_raster_path, polygon_shp, points_csv, output_folder_name)
    else:
        progress_label2.config(text="No raster found. Please run Step 1 first.")


# Function to check if the raster file exists
def check_raster_file_exists():
    output_folder_name = f"run_{date.today()}"
    output_raster_path = f"{output_folder_name}/{output_folder_name}_growth_forms_raster.tif"
    return os.path.exists(output_raster_path), output_raster_path

# Run section 2 only if the raster file exists
def run_section_2_only():
    raster_exists, raster_path = check_raster_file_exists()
    if raster_exists:
        # Load the raster and pass it to section 2
        with rasterio.open(raster_path) as src:
            classified_raster = src.read()
            classified_raster = np.moveaxis(classified_raster, 0, -1)  # Move the band axis to the last dimension
            transform = src.transform
            compute_growth_form_voronoi_polygons(classified_raster)
    else:
        progress_label2.config(text="No raster found. Please run Step 1 first.")

# Function to create mask for polygon-raster intersection
def polygon_raster_intersection_mask(polygon, raster_shape, transform):
    minx, miny, maxx, maxy = polygon.bounds
    # Get pixel coordinates of bounding box
    ulx, uly = transform * (minx, maxy)
    lrx, lry = transform * (maxx, miny)
    x, y = np.meshgrid(np.arange(ulx, lrx + 1), np.arange(uly, lry + 1))
    x, y = x.flatten(), y.flatten()
    # Convert pixel coordinates to geographic coordinates
    lon, lat = transform * (x, y)
    # Create points
    points = np.column_stack((lon, lat))
    # Create mask
    mask = [point.within(polygon) for point in map(shapely.geometry.Point, points)]
    mask = np.array(mask).reshape(raster_shape)
    return mask

def get_image_metadata(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        for tag, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name == "DateTimeOriginal":
                return value
    except Exception as e:
        progress_label3.config(text=f"Error extracting metadata for: {e}")
    return None

def perform_yolo_predictions_threaded():
    # Start YOLO prediction process in a separate thread
    threading.Thread(target=perform_yolo_predictions, args=(images_folder_entry.get(),)).start()

### Main function section 3
def perform_yolo_predictions(image_path):    
    progress_label3.config(text="Loading animal detection model...")

    # Create output folder with today's date
    output_folder_name = f"run_{date.today()}"
    os.makedirs(output_folder_name, exist_ok=True)

    # Construct the path to the model weights file
    model_path = os.path.join(script_dir, "yoloweights", "best.pt")

    # Load the model using the constructed path
    yolo_model = YOLO(model_path)
    
    progress_label3.config(text="Predicting images... Expect approximately 1 second processing time per image.")


    results = yolo_model.predict(source=image_path + "/*/*", # double wildcard since we have two folder levels
                        stream = True,
                        save = False #The images for a certain species can later on be retrieved via the paths in the csv.
                        #conf = 0.7
                        )
    
    # Get list of species names from the model
    species_names = list(yolo_model.names.values())

    # Initialize DataFrame to store results
    columns = ['Name', 'Date_Time'] + species_names
    df = pd.DataFrame(columns=columns)

    for r in results:
        try:
            # Extract date and time from image metadata
            date_time = get_image_metadata(r.path)
            if date_time is None:
                progress_label3.config(text=f"Metadata not found for image: {r.path}")
                continue

            # Initialize species detection dictionary
            species_detections = {species: 0 for species in species_names}
            conf = r.boxes.conf.numpy()
            for idx, t in enumerate(r.boxes.cls.numpy()):
                t = int(t)  # Convert to integer
                if conf[idx] > species_detections[species_names[t]]:
                    species_detections[species_names[t]] = conf[idx]

            # Create row for the current image
            row = [r.path, date_time] + [species_detections[species] for species in species_names]

            # Append row to DataFrame
            df.loc[len(df)] = row
        
        except FileNotFoundError as e:
            print(f"File not found during processing: {e.filename}. Skipping...")
            continue

    progress_label3.config(text="Predictions done. Saving results...")

    # Write DataFrame to CSV file
    csv_file = f"{output_folder_name}/{output_folder_name}_full_image_annotations.csv"
    df.to_csv(csv_file, index=False)
    
    progress_label3.config(text=f"Success. Predictions saved under {output_folder_name}/{output_folder_name}_full_image_annotations.csv.")
    messagebox.showinfo("Success", "Predictions saved.")


def browse_folder(entry_field):
    folder_path = filedialog.askdirectory()
    entry_field.delete(0, tk.END)
    entry_field.insert(0, folder_path)

## Main function section 4
def RAI():
    progress_label4.config(text="Computing relative abundance indices (RAIs) from detections and converting to feeding units...")

    # Create output folder with today's date
    output_folder_name = f"run_{date.today()}"
    os.makedirs(output_folder_name, exist_ok=True)

    # Construct the path to the model weights file
    model_path = os.path.join(script_dir, "yoloweights", "best.pt")

    # Load YOLO model
    yolo_model = YOLO(model_path)
    # Get list of species names from the model
    species_names = list(yolo_model.names.values())
    df = pd.read_csv(f"{output_folder_name}/{output_folder_name}_full_image_annotations.csv")

    # Define the threshold for probability
    threshold = float(prob_thresh_entry.get())

    # Convert Date_Time column to datetime type
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format="%Y:%m:%d %H:%M:%S")

    # Define a function to check if images are duplicates within a 25-minute window
    def is_duplicate(image_time, folder_times):
        for t in folder_times:
            if abs((image_time - t).total_seconds()) <= 1500: 
                return True
        return False

    # Create a dictionary to store folder-wise counts
    folder_counts = {}

    # Iterate over rows in the DataFrame
    for index, row in df.iterrows():
        # Extract folder name from the file path
        folder_path, _ = os.path.split(row['Name'])
        folder_name = os.path.basename(folder_path)
        
        # Initialize counts for the folder if it's not in the dictionary
        if folder_name not in folder_counts:
            folder_counts[folder_name] = {s: 0 for s in species_names}
            folder_counts[folder_name]['times'] = []
        
        # Check if the probability for each species is higher than the threshold
        for s in species_names:
            if row[s] > threshold:
                # Check if the image is a duplicate
                if not is_duplicate(row['Date_Time'], folder_counts[folder_name]['times']):
                    folder_counts[folder_name][s] += 1  # Increment count for the species
                    folder_counts[folder_name]['times'].append(row['Date_Time'])

    # Create a DataFrame from the dictionary
    results = pd.DataFrame.from_dict(folder_counts, orient='index').reset_index()
    results.rename(columns={'index': 'Folder'}, inplace=True)

    # Calculate total detections for each species within each site
    total_counts = results[species_names].sum()

    def parse_timestamp_string(timestamp_string):
        if isinstance(timestamp_string, list):
            timestamps = timestamp_string
        elif isinstance(timestamp_string, pd.Timestamp):
            timestamps = [timestamp_string]
        else:
            raise ValueError("Input must be either a list or a pandas Timestamp object")

        return [pd.to_datetime(ts) for ts in timestamps]

    results['parsed_timestamps'] = results['times'].apply(parse_timestamp_string)

    # Calculate time range for each row
    results['time_range_days'] = results['parsed_timestamps'].apply(lambda ts_list: (max(ts_list) - min(ts_list)).total_seconds() / (60 * 60 * 24))

    # Add columns for RAI of each species, RAI equivalent to capture events per day of camera trapping
    for s in species_names:
        results[f'Fraction_{s}'] = (results[s] / total_counts[s]) * 100
        results[f'RAI_{s}'] = (results[s]/ (results["time_range_days"]))

    # Reformat table
    results.rename(columns={'Folder': 'Site'}, inplace=True)
    results.drop(columns=['times', 'parsed_timestamps'], inplace=True)

    progress_label4.config(text="RAIs computed. Converting to feeding units...")    

    # Herbivory equivalents from Bothma et al. 2004
    data = {
        "Species": ["zebra", "wildebeest", "oryx", "eland", "elephant", "impala", "rhino", "giraffe", "kudu"],
        "Grazing unit": [1.32, 1, 1.1, 2, 9.8, 0.3, 3.1, 3.2, 0.8],
        "Browsing unit": [1.59, 1.21, 1.36, 2.4, 11.78, 0.4, 3.76, 3.8, 1]
    }
    df = pd.DataFrame(data)

    # Calculate grazing and browsing units for each species
    grazing_units = []
    browsing_units = []

    for col in results.columns:
        if col.startswith('RAI_'):
            species = col.split('_')[-1]
            grazing_unit = df.loc[df['Species'] == species, 'Grazing unit'].values
            browsing_unit = df.loc[df['Species'] == species, 'Browsing unit'].values
            if grazing_unit.size > 0:
                grazing_units.append(grazing_unit[0] * results[col])
            if browsing_unit.size > 0:
                browsing_units.append(browsing_unit[0] * results[col])

    results['Grazing_units_absolute'] = sum(grazing_units)
    results['Browsing_units_absolute'] = sum(browsing_units)

    # Save the updated results to a new CSV file
    csv_file = f"{output_folder_name}/{output_folder_name}_RAI.csv"
    results.to_csv(csv_file, index=False)
    progress_label4.config(text=f"Relative abundance and feeding indices saved to: {csv_file}")

def output():
    progress_label5.config(text="Computing final results and visualizations...")

    # Create output folder with today's date
    output_folder_name = f"run_{date.today()}"
    os.makedirs(output_folder_name, exist_ok=True)
    # File paths
    growth_form_file = f"{output_folder_name}/{output_folder_name}_growth_forms.csv"
    rai_file = f"{output_folder_name}/{output_folder_name}_RAI.csv"

    # Read CSV files
    growth_form_df = pd.read_csv(growth_form_file)
    rai_df = pd.read_csv(rai_file)

    merged_df = pd.merge(growth_form_df, rai_df[['Site', 'Grazing_units_absolute', 'Browsing_units_absolute']], on='Site', how='left')
    merged_df['Grazing_units_relative'] = merged_df['Grazing_units_absolute']/merged_df['area_hectares']
    merged_df['Browsing_units_relative'] = merged_df['Browsing_units_absolute']/merged_df['area_hectares']

    merged_df['Browsing_units/ha_woody_vegetation'] = merged_df['Browsing_units_absolute']/merged_df['area_hectares']/merged_df['mean_woody_fraction']
    merged_df['Grazing_units/ha_herbaceous_vegetation'] = merged_df['Grazing_units_absolute']/merged_df['area_hectares']/merged_df['mean_herbaceous_fraction']

    # Save updated DataFrame
    output_file = f"{output_folder_name}/{output_folder_name}_results.csv"
    merged_df.to_csv(output_file, index=False)
    progress_label5.config(text=f"Feeding units and growth forms results table saved to: {output_file}")

    ## Plots
    df = merged_df.dropna(subset=['Grazing_units/ha_herbaceous_vegetation'])
    sorted_df = df.sort_values(by='Grazing_units/ha_herbaceous_vegetation')

    # Create figure and axis objects
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot bars for mean_herbaceous_fraction on the first y-axis
    bar_width = 0.4
    bar_positions = np.arange(len(sorted_df))
    ax1.bar(bar_positions - bar_width/2, sorted_df['Grazing_units/ha_herbaceous_vegetation'], color='darkgreen', width=bar_width)

    # Set x-axis ticks and labels to correspond to the "Site"
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(sorted_df['Site'], rotation=45, ha='right')

    # Set labels for the axes
    ax1.set_xlabel('Site', labelpad=10)
    ax1.set_ylabel('Grazer units per camera day and hectare of herbaceous vegetation')

    # Adjust layout
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(f"{output_folder_name}/{output_folder_name}_grazer_units_barplot.png")

    sorted_df = df.sort_values(by='Browsing_units/ha_woody_vegetation')
    # Plot bars for mean_woody_fraction on the first y-axis
    ax1.bar(bar_positions - bar_width/2, sorted_df['Browsing_units/ha_woody_vegetation'], color='darkgreen', width=bar_width)

    # Set x-axis ticks and labels to correspond to the "Site"
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(df['Site'], rotation=45, ha='right')

    # Set labels for the axes
    ax1.set_xlabel('Site', labelpad=10)
    ax1.set_ylabel('Browser units per camera day and hectare of herbaceous vegetation')

    # Adjust layout
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(f"{output_folder_name}/{output_folder_name}_browser_units_barplot.png")

    progress_label5.config(text=f"Result plots saved to: {output_folder_name}/{output_folder_name}_browser/grazer_units_barplot.png")

# Create the main application window
app = tk.Tk()
app.title("Herbivory Monitoring Tool")
app.configure(bg="#ffffff")

# Create a Notebook (tabbed interface)
notebook = ttk.Notebook(app)
notebook.pack(fill='both', expand=True)

# Create individual frames for each section
gf_frame = ttk.Frame(notebook)
voronoi_frame = ttk.Frame(notebook)
images_frame = ttk.Frame(notebook)
RAI_frame = ttk.Frame(notebook)
output_frame = ttk.Frame(notebook)

# Add frames to the Notebook with respective titles
notebook.add(images_frame, text='Step 1: Image Classification')
notebook.add(RAI_frame, text='Step 2: RAIs')
notebook.add(gf_frame, text='Step 3: Growth Form Models')
notebook.add(voronoi_frame, text='Step 4: Voronoi Tessellation')
notebook.add(output_frame, text='Step 5: Results and Visualizations')

# Step 1: Growth form models

input_raster_label = tk.Label(gf_frame, text="Sentinel-2 Input Raster:")
input_raster_label.grid(row=0, column=0, padx=10, pady=5)

input_raster_entry = tk.Entry(gf_frame, width=50)
input_raster_entry.grid(row=0, column=1, padx=10, pady=5)

input_raster_button = tk.Button(gf_frame, text="Browse", command=lambda: browse_file(input_raster_entry))
input_raster_button.grid(row=0, column=2, padx=10, pady=5)

def start_classification_thread():
    input_raster_path = input_raster_entry.get()
    
    # Start classification process in a separate thread
    threading.Thread(target=classify_raster, args=(input_raster_path, progress_label)).start()

run_gf_button = tk.Button(gf_frame, text="Run Growth Form Model", command=start_classification_thread)
run_gf_button.grid(row=1, column=0, columnspan=3, padx=10, pady=5)

progress_label = tk.Label(gf_frame, text="")
progress_label.grid(row=2, column=0, columnspan=3, padx=10, pady=5)

information_1 = tk.Label(gf_frame, text="Please provide a Sentinel-2 raster that corresponds to the period during which the images were taken. Cloud coverage can degrade algorithm performance.")
information_1.grid(row=3, column=0, columnspan=3, padx=10, pady=5)

image_path1 = os.path.join(script_dir, "app_figures", "growth_form_models.jpg")
image1 = Image.open(image_path1)
photo1 = ImageTk.PhotoImage(image1)
image_label1 = tk.Label(gf_frame, image=photo1)
image_label1.grid(row=5, column=0, columnspan=3, padx=10, pady=5)

# Step 2: Voronoi tessellation

polygon_shp_label = tk.Label(voronoi_frame, text="Shapefile of the Protected Area:")
polygon_shp_label.grid(row=0, column=0, padx=10, pady=5)

polygon_shp_entry = tk.Entry(voronoi_frame, width=50)
polygon_shp_entry.grid(row=0, column=1, padx=10, pady=5)

polygon_shp_button = tk.Button(voronoi_frame, text="Browse", command=lambda: browse_file(polygon_shp_entry))
polygon_shp_button.grid(row=0, column=2, padx=10, pady=5)

points_csv_label = tk.Label(voronoi_frame, text="CSV file with waterpoint coordinates:")
points_csv_label.grid(row=1, column=0, padx=10, pady=5)

points_csv_entry = tk.Entry(voronoi_frame, width=50)
points_csv_entry.grid(row=1, column=1, padx=10, pady=5)

points_csv_button = tk.Button(voronoi_frame, text="Browse", command=lambda: browse_file(points_csv_entry))
points_csv_button.grid(row=1, column=2, padx=10, pady=5)

run_voronoi_button = tk.Button(voronoi_frame, text="Run Voronoi Computation", command=run_voronoi_button_callback)
run_voronoi_button.grid(row=2, column=0, columnspan=3, padx=10, pady=5)

progress_label2 = tk.Label(voronoi_frame, text="")
progress_label2.grid(row=3, column=0, columnspan=3, padx=10, pady=5)

information_2 = tk.Label(voronoi_frame, text="The shapefile (.shp) must be in EPSG:4326 and accompanied by .cpg, .dbf, .prj and .shx files. The .csv should contain decimal coordinates in columns named Longitude and Latitude.")
information_2.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

image_path2 = os.path.join(script_dir, "app_figures", "voronoi_tessellation.jpg")
image2 = Image.open(image_path2)
photo2 = ImageTk.PhotoImage(image2)
image_label2 = tk.Label(voronoi_frame, image=photo2)
image_label2.grid(row=5, column=0, columnspan=3, padx=10, pady=5)

# Step 3: Image classification

images_folder_label = tk.Label(images_frame, text="Folder containing camera trap images:")
images_folder_label.grid(row=0, column=0, padx=10, pady=5)

images_folder_entry = tk.Entry(images_frame, width=50)
images_folder_entry.grid(row=0, column=1, padx=10, pady=5)

images_folder_button = tk.Button(images_frame, text="Browse", command=lambda: browse_folder(images_folder_entry))
images_folder_button.grid(row=0, column=2, padx=10, pady=5)

perform_yolo_button = tk.Button(images_frame, text="Perform YOLOv8 Predictions", command=perform_yolo_predictions_threaded)
perform_yolo_button.grid(row=1, column=0, columnspan=3, padx=10, pady=5)

progress_label3 = tk.Label(images_frame, text="")
progress_label3.grid(row=2, column=0, columnspan=3, padx=10, pady=5)

information_3 = tk.Label(images_frame, text="Please provide the path to a directory with folders that correspond to waterpoints. Only images in waterpoint folders will be classified.")
information_3.grid(row=3, column=0, columnspan=3, padx=10, pady=5)

image_path3 = os.path.join(script_dir, "app_figures", "image_classification.jpg")
image3 = Image.open(image_path3)
photo3 = ImageTk.PhotoImage(image3)
image_label3 = tk.Label(images_frame, image=photo3)
image_label3.grid(row=5, column=0, columnspan=3, padx=10, pady=5)

# Step 4: RAIs

prob_thresh = tk.Label(RAI_frame, text="Detection probability threshold (default: 0.5):")
prob_thresh.grid(row=0, column=0, padx=10, pady=5)

prob_thresh_entry = tk.Entry(RAI_frame, width=10)
prob_thresh_entry.grid(row=0, column=1, padx=10, pady=5)

perform_RAI_button = tk.Button(RAI_frame, text = "Perform RAI Analysis", command = RAI)
perform_RAI_button.grid(row = 1, column = 1, columnspan = 3, padx = 10, pady = 5)

progress_label4 = tk.Label(RAI_frame, text="")
progress_label4.grid(row=2, column=0, columnspan=3, padx=10, pady=5)

information_4 = tk.Label(RAI_frame, text="This converts the detections to RAIs [number of drinking events per day of camera trapping], then to feeding units based on Bothma et al. (2004).")
information_4.grid(row=3, column=0, columnspan=3, padx=10, pady=5)

information_42 = tk.Label(RAI_frame, text="All cameras at the same site need to have their time synchronized for duplicates to be effectively excluded. Classification performance should be assessed before setting the threshold.")
information_42.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

image_path4 = os.path.join(script_dir, "app_figures", "RAIs.jpg")
image4 = Image.open(image_path4)
photo4 = ImageTk.PhotoImage(image4)
image_label4 = tk.Label(RAI_frame, image=photo4)
image_label4.grid(row=5, column=0, columnspan=3, padx=10, pady=5)

# Step 5: Results
run_gf_button = tk.Button(output_frame, text="Compute Results", command=output)
run_gf_button.grid(row=2, column=1, columnspan=3, padx=10, pady=5)

progress_label5 = tk.Label(output_frame, text="")
progress_label5.grid(row=3, column=0, columnspan=3, padx=10, pady=5)

information_5 = tk.Label(output_frame, text="This step unites the growth form predictions with the stocking rate estimates. All outputs are stored in a folder named after today's date created where the app is saved.")
information_5.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

image_path5 = os.path.join(script_dir, "app_figures", "results_and_visualizations.jpg")
image5 = Image.open(image_path5)
photo5 = ImageTk.PhotoImage(image5)
image_label5 = tk.Label(output_frame, image=photo5)
image_label5.grid(row=5, column=0, columnspan=3, padx=10, pady=5)

# Run the application
app.mainloop()