########################################################################################################################################################################################
### Spatially and Temporally Explicit Herbivory Monitoring in Water-limited Savannas
### Desktop Application
### Manuel Weber
### https://github.com/Manuel-Weber-ETH/cameratraps.git
########################################################################################################################################################################################
### Structure:
### 1. Loading libraries ____________________________________________________________________________________________________
### 2. Setting global variables ____________________________________________________________________________________________________
### 3. Helper functions to browse files and folders ____________________________________________________________________________________________________
### 4. Processing of the Sentinel-2 raster ____________________________________________________________________________________________________
### 5. Voronoi tessellation ____________________________________________________________________________________________________
### 6. Camera trap images processing ____________________________________________________________________________________________________
### 7. Relative abundance indices ____________________________________________________________________________________________________
### 8. Producing visualizations ____________________________________________________________________________________________________
### 9. Merging vegetation categories and RAIs ____________________________________________________________________________________________________
### 10. Download GEE script ____________________________________________________________________________________________________
### 11. Graphical user interface: Setup and style ____________________________________________________________________________________________________
### 12. Tab 1: Camera trap images ____________________________________________________________________________________________________
### 13. Tab 2: Vegetation categories ____________________________________________________________________________________________________
########################################################################################################################################################################################


### 1. Loading libraries ____________________________________________________________________________________________________
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import date, timedelta, datetime
import os
import sys
import glob
import numpy as np
import rasterio
from tensorflow.keras.models import load_model
import time
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from PIL import Image, ExifTags, ImageTk
from rasterio.mask import geometry_mask
from rasterio.windows import Window
import shapely
from scipy.spatial import Voronoi
from ultralytics import YOLO, YOLOv10
import threading
from tkinter import ttk,Toplevel, Label
import shutil
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import rasterio.plot
from rasterio.mask import mask
from matplotlib.patches import Patch

### 2. Setting global variables ____________________________________________________________________________________________________
script_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
output_base_path = ""
global results_df
results_df = pd.DataFrame()
global checkbox_vars
checkbox_vars = {}

### 3. Helper functions to browse files and folders ____________________________________________________________________________________________________
def browse_file(entry_field, message, file_types):
    messagebox.showinfo("Information", message)
    filepath = filedialog.askopenfilename(filetypes=file_types)
    if filepath:
        entry_field.delete(0, tk.END)
        entry_field.insert(0, filepath)
    else:
        messagebox.showerror("Error", "No file selected.")

def browse_file_raster(entry_field):
    browse_file(entry_field, "Please provide a Sentinel-2 1C raster that corresponds to the period during which the images were taken. Cloud coverage can degrade algorithm performance.", [("TIF files", "*.tif")])

def browse_file_shp(entry_field):
    browse_file(entry_field, "The shapefile (.shp) must be in EPSG:4326 and accompanied by .cpg, .dbf, .prj and .shx files.", [("Shapefiles", "*.shp")])

def browse_file_csv(entry_field):
    browse_file(entry_field, "The .csv should contain decimal coordinates in columns named 'Longitude' and 'Latitude'. The site names should be stored in a column named 'Site'.", [("CSV files", "*.csv")])


def browse_and_create_output_directory():
    global output_base_path
    base_dir = filedialog.askdirectory()
    if not base_dir:
        messagebox.showerror("Error", "No directory selected.")
        return
    
    today = date.today()
    session_dir_name = f"run_{today}"
    full_path = os.path.join(base_dir, session_dir_name)
    
    try:
        os.makedirs(full_path, exist_ok=True)
        output_base_path = full_path
        messagebox.showinfo("Success", f"Output will be saved in: {output_base_path}")
    except PermissionError:
        messagebox.showerror("Permission Denied", "You do not have permission to create a directory in the selected location.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create directory: {e}")

def browse_folder(entry_field):
    messagebox.showinfo("Information", "Please provide the path to a directory with folders that correspond to waterpoints. Only images in waterpoint folders will be classified.")
    folder_path = filedialog.askdirectory()
    entry_field.delete(0, tk.END)
    entry_field.insert(0, folder_path)

### 4. Processing of the Sentinel-2 raster ____________________________________________________________________________________________________
def raster_to_array(raster_path):
    try:
        with rasterio.open(raster_path) as src:
            raster_array = src.read()
            raster_array = np.moveaxis(raster_array, 0, -1)  # Move the band axis to the last dimension
        return raster_array
    except rasterio.errors.RasterioIOError:
        messagebox.showerror("Error", "Failed to read the raster file. Please check the file path and try again.")
        return None

def apply_model_to_raster(gf_model, raster_array):
    if raster_array is None:
        return None

    height, width, bands = raster_array.shape
    flattened_raster = raster_array.reshape(-1, bands)
    predictions = gf_model.predict(flattened_raster)
    output = predictions.reshape(height, width, 3)
    return output

def run_vegetation_category_model():
    input_raster_path = input_raster_entry.get()
    if not os.path.exists(input_raster_path):
        messagebox.showerror("Error", "Input raster file not found.")
        return
    # Start the classification in a separate thread
    threading.Thread(target=classify_raster, args=(input_raster_path, progress_label2)).start()

def classify_raster(input_raster_path, progress_label2):
    global output_base_path
    if not output_base_path:
        messagebox.showerror("Error", "Output directory not set.")
        return
    
    progress_label2.config(text="Loading model...")

    model_path = os.path.join(script_dir, "growthformDLmodel")
    
    try:
        gf_model = load_model(model_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load the model: {e}")
        return
    
    raster_array = raster_to_array(input_raster_path)
    if raster_array is None:
        return
    
    progress_label2.config(text="Raster normalization...")
    
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

    progress_label2.config(text="Applying model... Please allow up to 1 min per 1000 ha.")

    output_raster_array = apply_model_to_raster(gf_model, raster_array)
    if output_raster_array is None:
        return

    progress_label2.config(text="Success. Saving output raster...")
    
    output_raster_path = f"{output_base_path}/vegetation_categories_raster.tif"
    
    with rasterio.open(input_raster_path) as src:
        profile = src.profile
    
    profile.update(
        count=3,  # Update the number of bands to 3
        dtype=output_raster_array.dtype
    )
    
    try:
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            for band_index in range(output_raster_array.shape[2]):
                dst.write(output_raster_array[:, :, band_index], band_index + 1)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save the output raster: {e}")
        return
    
    progress_label2.config(text=f"vegetation category raster saved at: {output_raster_path}")
    messagebox.showinfo("Success", "vegetation category raster saved successfully.")


# Function to check if the raster file exists
def check_raster_file_exists():
    global output_base_path
    if not output_base_path:
        messagebox.showerror("Error", "Output directory not set.")
        return
    output_raster_path = f"{output_base_path}/vegetation_categories_raster.tif"
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
            compute_vegetation_category_voronoi_polygons(classified_raster)
    else:
        progress_label2.config(text="No raster found. Please run Step 3 first.")

### 5. Voronoi tessellation ____________________________________________________________________________________________________
voronoi_function_running = False
def compute_vegetation_category_voronoi_polygons(output_raster_path, polygon_shp, points_csv, output_base_path):
    # Update progress label
    progress_label2.config(text="Loading vegetation category raster...")

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
    progress_label2.config(text="Computing vegetation category fractions for Voronoi polygons...")

    # Overlay Voronoi polygons with input polygon to get intersection
    result = gpd.overlay(df1=voronois, df2=polygon_gdf, how="intersection")

    # Initialize lists to store mean vegetation category fractions for each band
    mean_band1_fractions = []
    mean_band2_fractions = []
    mean_band3_fractions = []
    area_hectares = []

    # Convert the GeoDataFrame to EPSG:3857
    # Note that I am not using a UTM crs, in a future version we may need to add the option to choose the crs since the area calculations will be different
    voronois_3857 = voronois.to_crs(epsg=3857)
    # Save Voronoi polygons as a shapefile
    voronois_3857.to_file(f"{output_base_path}/voronoi_polygons.shp")

    # Iterate over each Voronoi polygon
    for index, row in result.iterrows():
        polygon_3857 = voronois_3857.iloc[index]['geometry']
        # Compute the area of the polygon in hectares
        area_hectares.append(polygon_3857.area / 10000)
        polygon = row['geometry']
        # Your existing code for extracting pixel values and calculating mean vegetation category fractions
        mask = geometry_mask([polygon], out_shape=classified_raster.shape[:2], transform=transform, invert=True)
        masked_values = classified_raster[mask]
        mean_band1 = np.mean(masked_values[:, 0])
        mean_band2 = np.mean(masked_values[:, 1])
        mean_band3 = np.mean(masked_values[:, 2])
        mean_band1_fractions.append(mean_band1)
        mean_band2_fractions.append(mean_band2)
        mean_band3_fractions.append(mean_band3)

    # Add mean vegetation category fractions to the GeoDataFrame
    result['Site'] = points_df['Site']
    result['mean_woody_fraction'] = mean_band1_fractions
    result['mean_herbaceous_fraction'] = mean_band2_fractions
    result['mean_bare_fraction'] = mean_band3_fractions
    result.drop(columns=['id', 'geometry'], inplace=True)
    # Add the area values as a new column in the DataFrame
    result['area_hectares'] = area_hectares

    # Save results to a CSV file
    output_csv = f"{output_base_path}/vegetation_categories.csv"
    result.to_csv(output_csv)

    # Update progress label
    progress_label2.config(text=f"Mean vegetation category fractions saved to: {output_csv}")
    
    # Show success message
    messagebox.showinfo("Success", "vegetation category fractions saved successfully.")

def run_voronoi_button_callback():
    # Define input variables (will be replaced with values from GUI)
    global output_base_path
    if not output_base_path:
        messagebox.showerror("Error", "Output directory not set.")
        return
    output_raster_path = f"{output_base_path}/vegetation_categories_raster.tif"
    polygon_shp = polygon_shp_entry.get()
    points_csv = points_csv_entry.get()

    # Check if classified raster exists
    if os.path.exists(output_raster_path):
        # Run the function
        compute_vegetation_category_voronoi_polygons(output_raster_path, polygon_shp, points_csv, output_base_path)
        progress_label2.config(text=f"compute_vegetation_category_voronoi_polygons successfully run. Preparing to run update_class_raster_voronoi_plot")
        # Update the first plot canvas
        update_class_raster_voronoi_plot(output_raster_path, polygon_shp, points_csv)
        progress_label2.config(text=f"update_class_raster_voronoi_plot successfully run.")
    else:
        progress_label2.config(text="No raster found. Please run Step 3 first.")
    # Run the output function
    output()

def update_class_raster_voronoi_plot(raster_path, polygon_shp, points_csv):
    global class_raster_voronoi_canvas
    
    # Load classified raster
    with rasterio.open(raster_path) as src:
        classified_raster = src.read()
        classified_raster = np.moveaxis(classified_raster, 0, -1)  # Move the band axis to the last dimension
        transform = src.transform

    # Load polygon from shapefile
    polygon_gdf = gpd.read_file(polygon_shp)
    polygon = polygon_gdf.geometry.iloc[0]

    # Mask the raster with the polygon
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, [polygon], crop=True)
        out_image = out_image.astype(np.float32)
        out_image /= np.max(out_image, axis=(1, 2), keepdims=True)
        out_image = np.moveaxis(out_image, 0, -1)  # Move bands to last dimension

    # Load points from CSV
    points_df = pd.read_csv(points_csv)
    points_gdf = gpd.GeoDataFrame(points_df, geometry=gpd.points_from_xy(points_df['Longitude'], points_df['Latitude']), crs='EPSG:4326').to_crs(polygon_gdf.crs)

    # Load Voronoi polygons from shapefile
    voronois_gdf = gpd.read_file(f"{output_base_path}/voronoi_polygons.shp").to_crs(polygon_gdf.crs)

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figsize to control the size

    # Calculate extents
    extent = [out_transform[2], out_transform[2] + out_image.shape[1] * out_transform[0],
              out_transform[5] + out_image.shape[0] * out_transform[4], out_transform[5]]

    # Plot raster
    ax.imshow(out_image, extent=extent, origin='upper')

    # Plot Voronoi polygons
    voronois_gdf.boundary.plot(ax=ax, color='white', linewidth=1, label='Voronoi Polygons')

    # Plot waterpoints
    points_gdf.plot(ax=ax, color='lightblue', markersize=30, marker='o', label='Waterpoints')

    # Add labels to waterpoints
    for x, y, label in zip(points_gdf.geometry.x, points_gdf.geometry.y, points_gdf['Site']):
        ax.text(x, y, label, fontsize=8, ha='right', va='bottom', color='white')

    # Add custom legend
    legend_elements = [
        Patch(facecolor='red', edgecolor='none', label='Woody Vegetation'),
        Patch(facecolor='turquoise', edgecolor='none', label='Herbaceous Vegetation')
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize='small', ncol = 2)

    # Reduce the size of coordinate labels along the axis
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Adjust layout to fit better
    plt.tight_layout()

    # Update the canvas
    class_raster_voronoi_canvas.figure = fig
    class_raster_voronoi_canvas.draw()

    # Save the figure
    save_path = os.path.join(output_base_path, 'map.png')
    plt.savefig(save_path)

    plt.close(fig)

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

### 6. Camera trap images processing ____________________________________________________________________________________________________
def check_and_remove_corrupted_files(directory_path, progress_label1):
    count = 0
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:  # Open and check the image
                        img.verify()  # Verify the image is intact
                except (IOError, SyntaxError) as e:
                    progress_label1.config(text=f"Corrupted file detected and removed: {file_path}")
                    os.remove(file_path)  # Remove the corrupted file
                    count += 1
    progress_label1.config(text=f"Total corrupted files removed: {count}")


def get_image_metadata(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        for tag, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            if tag_name == "DateTimeOriginal":
                return value
    except Exception as e:
        progress_label1.config(text=f"Error extracting metadata for: {e}")
    return None


def perform_yolo_predictions_threaded():
    threading.Thread(target=perform_yolo_predictions, args=(images_folder_entry.get(), progress_label1,)).start()


def perform_yolo_predictions(image_path, progress_label1):
    global results_df 

    if not output_base_path:
        messagebox.showerror("Error", "Output directory not set.")
        return

    if not os.path.exists(image_path):
        messagebox.showerror("Error", "Image directory not found.")
        return

    progress_label1.after(0, lambda: progress_label1.config(text="Verifying images..."))

    model_path = os.path.join(script_dir, "yoloweightsv10.pt")
    
    try:
        yolo_model = YOLOv10(model_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
        return

    all_image_paths = glob.glob(os.path.join(image_path, '*/*'))
    if not all_image_paths:
        messagebox.showerror("Error", "No images found in the specified directory.")
        return

    batch_size = 100
    total_batches = (len(all_image_paths) + batch_size - 1) // batch_size

    temp_results = []  # Use a temporary list to compile results

    threshold = float(prob_thresh_entry.get()) if prob_thresh_entry.get() else 0.75


    for batch_index in range(total_batches):
        batch_start = batch_index * batch_size
        batch_end = min((batch_index + 1) * batch_size, len(all_image_paths))
        current_batch_paths = all_image_paths[batch_start:batch_end]

        progress_label1.config(text=f"Classifying batch {batch_index + 1} of {total_batches}...")
        results = yolo_model.predict(source=current_batch_paths, stream=True, save=False, conf=threshold)

        for r in results:
            max_confidences = {species: 0 for species in yolo_model.names.values()}
            date_time = get_image_metadata(r.path)
            if date_time is None:
                progress_label1.config(text=f"Metadata not found for image: {r.path}")
                continue

            for idx, class_id in enumerate(r.boxes.cls.numpy()):
                species = yolo_model.names[int(class_id)]
                confidence = r.boxes.conf.numpy()[idx]
                if confidence > max_confidences[species]:
                    max_confidences[species] = confidence

            max_confidences['Name'] = r.path
            max_confidences['Date_Time'] = date_time
            temp_results.append(max_confidences)

    columns = ['Name', 'Date_Time'] + list(yolo_model.names.values())
    results_df = pd.DataFrame(temp_results, columns=columns)

    species_image_counts = (results_df[list(yolo_model.names.values())] > 0).sum().to_dict()
    update_species_count_table(species_image_counts)

    progress_label1.config(text="All batches classified.")

    csv_file = f"{output_base_path}/full_image_annotations.csv"
    results_df.to_csv(csv_file, index=False)
    
    progress_label1.config(text=f"Success. Predictions saved under {output_base_path}/full_image_annotations.csv.")
    messagebox.showinfo("Success", "Predictions saved. Select species in table to sort images into folders.")


def update_species_count_table(species_counts):
    for i in detection_table.get_children():
        detection_table.delete(i)
    
    checkbox_vars.clear()

    for species, count in species_counts.items():
        cb_var = tk.BooleanVar(value=True)
        checkbox_vars[species] = cb_var
        detection_table.insert('', 'end', values=(species, count, '✓' if cb_var.get() else ''), tags=('checkbox',))


def toggle_checkbox(event):
    region = detection_table.identify("region", event.x, event.y)
    if region == "cell":
        row_id = detection_table.identify_row(event.y)
        column = detection_table.identify_column(event.x)
        if column == "#3":  # Ensure this is the checkbox column
            species = detection_table.item(row_id)['values'][0]
            checkbox_vars[species].set(not checkbox_vars[species].get())
            detection_table.item(row_id, values=(detection_table.item(row_id)['values'][0],
                                                 detection_table.item(row_id)['values'][1],
                                                 '✓' if checkbox_vars[species].get() else ''))

def organize_images():
    global results_df

    if results_df.empty:
        messagebox.showerror("Error", "No classification results found. Perform classification first.")
        return

    threshold = float(prob_thresh_entry.get()) if prob_thresh_entry.get() else 0.75
    selected_species = [species for species, var in checkbox_vars.items() if var.get()]

    if not selected_species:
        messagebox.showerror("Error", "No species selected.")
        return

    for species in selected_species:
        species_folder = os.path.join(output_base_path, species)
        if not os.path.exists(species_folder):
            os.makedirs(species_folder)

        filtered_df = results_df[results_df[species] >= threshold]

        for image_path in filtered_df['Name']:
            try:
                shutil.copy(image_path, species_folder)
            except Exception as e:
                progress_label2.config(text=f"Error moving image {image_path}: {str(e)}")

    messagebox.showinfo("Success", "Images have been organized into species folders.")

### 7. Relative abundance indices ____________________________________________________________________________________________________
def RAI(figure_canvas):
    messagebox.showinfo("Warning", "All cameras at the same site need to have their time synchronized for duplicates to be effectively excluded.")
    progress_label1.config(text="Computing relative abundance indices (RAIs) from detections and converting to feeding units...")

    global output_base_path
    if not output_base_path:
        messagebox.showerror("Error", "Output directory not set.")
        return

    model_path = os.path.join(script_dir, "yoloweightsv10.pt")
    yolo_model = YOLOv10(model_path)
    # yolo_model = YOLO(model_path) # For YOLOv8 model
    species_names = list(yolo_model.names.values())

    df = pd.read_csv(f"{output_base_path}/full_image_annotations.csv")
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], format="%Y:%m:%d %H:%M:%S")
    df['Folder'] = df['Name'].apply(lambda x: os.path.basename(os.path.dirname(x)))

    threshold = float(prob_thresh_entry.get()) if prob_thresh_entry.get() else 0.75

    def is_duplicate(image_time, folder_data, species):
        event_times = folder_data[species]['event_times']
        if not event_times:
            return False
        
        last_event_time = event_times[-1]
        time_difference_condition = abs((image_time - last_event_time).total_seconds()) <= 1500
        
        last_event_index = folder_data['all_times'].index(last_event_time)
        current_index = folder_data['all_times'].index(image_time)
        intervening_images = folder_data['all_times'][last_event_index + 1:current_index]
        different_species_count = sum(1 for time in intervening_images if any(folder_data[other_species]['detecting_images'].get(time, False) for other_species in species_names if other_species != species))
        
        different_species_condition = different_species_count < 3

        return time_difference_condition or different_species_condition

    folder_counts = {}
    for index, row in df.iterrows():
        folder_name = row['Folder']
        if folder_name not in folder_counts:
            folder_counts[folder_name] = {s: {'count': 0, 'event_times': [], 'detecting_images': {}} for s in species_names}
            folder_counts[folder_name]['all_times'] = []
        
        folder_counts[folder_name]['all_times'].append(row['Date_Time'])
        
        for s in species_names:
            if row[s] > threshold:
                folder_counts[folder_name][s]['detecting_images'][row['Date_Time']] = True

    for folder_name, folder_data in folder_counts.items():
        folder_data['all_times'].sort()
        
        for index, row in df[df['Folder'] == folder_name].iterrows():
            for s in species_names:
                if row[s] > threshold:
                    if not is_duplicate(row['Date_Time'], folder_data, s):
                        folder_data[s]['count'] += 1
                        folder_data[s]['event_times'].append(row['Date_Time'])

    final_counts = {folder_name: {s: data[s]['count'] for s in species_names} for folder_name, data in folder_counts.items()}
    results = pd.DataFrame.from_dict(final_counts, orient='index').reset_index()
    results.rename(columns={'index': 'Site'}, inplace=True)

    for folder_name in folder_counts:
        min_time = min(folder_counts[folder_name]['all_times'])
        max_time = max(folder_counts[folder_name]['all_times'])
        folder_counts[folder_name]['time_range_days'] = (max_time - min_time).total_seconds() / (60 * 60 * 24)

    results['time_range_days'] = results['Site'].apply(lambda x: folder_counts[x]['time_range_days'] if x in folder_counts else 0)

    for s in species_names:
        results[f'RAI_{s}'] = results.apply(
            lambda row: row[s] / row['time_range_days'] if row['time_range_days'] > 0 else 0,
            axis=1
        )

    herbivory_data = {
        "Species": ["zebra", "wildebeest", "oryx", "eland", "elephant", "impala", "rhino", "giraffe", "kudu", "springbok"],
        "Grazing unit": [1.32, 1, 1.1, 2, 9.8, 0.3, 3.1, 3.2, 0.8, 0.3],
        "Browsing unit": [1.59, 1.21, 1.36, 2.4, 11.78, 0.4, 3.76, 3.8, 1, 0.37],
        "Selectivity": [0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    }
    herbivory_df = pd.DataFrame(herbivory_data)

    results['Grazing_units_absolute_selective'] = 0.0
    results['Browsing_units_absolute_selective'] = 0.0
    results['Grazing_units_absolute_bulk'] = 0.0
    results['Browsing_units_absolute_bulk'] = 0.0

    for col in results.columns:
        if col.startswith('RAI_'):
            species = col.split('_')[-1]
            grazing_unit = herbivory_df.loc[herbivory_df['Species'] == species, 'Grazing unit'].values
            browsing_unit = herbivory_df.loc[herbivory_df['Species'] == species, 'Browsing unit'].values

            if grazing_unit.size > 0:
                if herbivory_df.loc[herbivory_df['Species'] == species, 'Selectivity'].values == 1:
                    results['Grazing_units_absolute_selective'] += grazing_unit[0] * results[col].fillna(0)
                else:
                    results['Grazing_units_absolute_bulk'] += grazing_unit[0] * results[col].fillna(0)
            if browsing_unit.size > 0:
                if herbivory_df.loc[herbivory_df['Species'] == species, 'Selectivity'].values == 1:
                    results['Browsing_units_absolute_selective'] += browsing_unit[0] * results[col].fillna(0)
                else:
                    results['Browsing_units_absolute_bulk'] += browsing_unit[0] * results[col].fillna(0)

    results.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    csv_file = f"{output_base_path}/RAI.csv"
    results.to_csv(csv_file, index=False)
    progress_label1.config(text=f"Relative abundance and feeding indices saved to: {csv_file}")
    absolute_feeding_units_plot(results, output_base_path, figure_canvas)

### 8. Producing visualizations ____________________________________________________________________________________________________
def absolute_feeding_units_plot(results, output_base_path, figure_canvas):
    sorted_results = results.sort_values(by='Site')
    
    fig, ax = plt.subplots(figsize=(6, 4)) 
    colors = {
        'Selective Grazing': '#2ca25f',
        'Bulk Grazing': '#99d8c9',
        'Selective Browsing': '#8c6c4f',
        'Bulk Browsing': '#c9ae91'
    }
    bar_width = 0.6
    y_positions = np.arange(len(sorted_results))

    for i, (index, row) in enumerate(sorted_results.iterrows()):
        ax.barh(i, row['Grazing_units_absolute_selective'], color=colors['Selective Grazing'], edgecolor='white', height=bar_width)
        ax.barh(i, row['Grazing_units_absolute_bulk'], left=row['Grazing_units_absolute_selective'], color=colors['Bulk Grazing'], edgecolor='white', height=bar_width)
        ax.barh(i, -row['Browsing_units_absolute_selective'], color=colors['Selective Browsing'], edgecolor='white', height=bar_width)
        ax.barh(i, -row['Browsing_units_absolute_bulk'], left=-row['Browsing_units_absolute_selective'], color=colors['Bulk Browsing'], edgecolor='white', height=bar_width)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_results['Site'])
    ax.set_xlabel('Feeding units detected per camera trap day')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(int(x))}"))  # Remove "-" and round the labels
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.15)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

    legend_elements = [
        Patch(facecolor=colors['Bulk Browsing'], edgecolor='none', label='Bulk Browsing'),
        Patch(facecolor=colors['Selective Browsing'], edgecolor='none', label='Selective Browsing'),
        Patch(facecolor=colors['Selective Grazing'], edgecolor='none', label='Selective Grazing'),
        Patch(facecolor=colors['Bulk Grazing'], edgecolor='none', label='Bulk Grazing')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize='small',  handlelength=1, handleheight=1, markerscale=0.8)

    plot_path = os.path.join(output_base_path, 'absolute_feeding_units.png')
    plt.savefig(plot_path)
    
    figure_canvas.figure = fig
    figure_canvas.draw()

    plt.close(fig)

def relative_feeding_units_plot():
    global feeding_units_canvas
    
    results = pd.read_csv(f"{output_base_path}/results.csv")

    results = results.dropna(subset=[
        'Grazing_units/ha_herbaceous_vegetation_selective', 
        'Browsing_units/ha_woody_vegetation_selective', 
        'Grazing_units/ha_herbaceous_vegetation_bulk', 
        'Browsing_units/ha_woody_vegetation_bulk'
    ])

    # Sort results by site name alphabetically
    sorted_results = results.sort_values(by='Site')

    # Plot the data
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {
        'Selective Grazing': '#2ca25f',
        'Bulk Grazing': '#99d8c9',
        'Selective Browsing': '#8c6c4f',
        'Bulk Browsing': '#c9ae91'
    }
    bar_width = 0.6
    y_positions = np.arange(len(sorted_results))

    for i, (index, row) in enumerate(sorted_results.iterrows()):
        ax.barh(i, row['Grazing_units/ha_herbaceous_vegetation_selective'], color=colors['Selective Grazing'], edgecolor='white', height=bar_width)
        ax.barh(i, row['Grazing_units/ha_herbaceous_vegetation_bulk'], left=row['Grazing_units/ha_herbaceous_vegetation_selective'], color=colors['Bulk Grazing'], edgecolor='white', height=bar_width)
        ax.barh(i, -row['Browsing_units/ha_woody_vegetation_selective'], color=colors['Selective Browsing'], edgecolor='white', height=bar_width)
        ax.barh(i, -row['Browsing_units/ha_woody_vegetation_bulk'], left=-row['Browsing_units/ha_woody_vegetation_selective'], color=colors['Bulk Browsing'], edgecolor='white', height=bar_width)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_results['Site'])
    ax.set_xlabel('Feeding units per camera trap day and ha of woody or herbaceous vegetation')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):.2f}"))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.15)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.7)

    legend_elements = [
        Patch(facecolor=colors['Bulk Browsing'], edgecolor='none', label='Bulk Browsing'),
        Patch(facecolor=colors['Selective Browsing'], edgecolor='none', label='Selective Browsing'),
        Patch(facecolor=colors['Selective Grazing'], edgecolor='none', label='Selective Grazing'),
        Patch(facecolor=colors['Bulk Grazing'], edgecolor='none', label='Bulk Grazing')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize='small',  handlelength=1, handleheight=1, markerscale=0.8)

    plot_path = os.path.join(output_base_path, 'relative_feeding_units_per_area_of_biomass.png')
    plt.savefig(plot_path)
    
    feeding_units_canvas.figure = fig
    feeding_units_canvas.draw()

    plt.close(fig)

### 9. Merging vegetation categories and RAIs ____________________________________________________________________________________________________
def output():
    progress_label2.config(text="Computing final results and visualizations...")

    global output_base_path
    if not output_base_path:
        messagebox.showerror("Error", "Output directory not set.")
        return
    
    # File paths
    vegetation_category_file = f"{output_base_path}/vegetation_categories.csv"
    rai_file = f"{output_base_path}/RAI.csv"

    # Read CSV files
    vegetation_category_df = pd.read_csv(vegetation_category_file)
    rai_df = pd.read_csv(rai_file)

    merged_df = pd.merge(vegetation_category_df, rai_df[['Site', 'Grazing_units_absolute_selective', 'Browsing_units_absolute_selective', 'Grazing_units_absolute_bulk', 'Browsing_units_absolute_bulk']], on='Site', how='left')
    merged_df['Grazing_units_relative_selective'] = merged_df['Grazing_units_absolute_selective'] / merged_df['area_hectares']
    merged_df['Browsing_units_relative_selective'] = merged_df['Browsing_units_absolute_selective'] / merged_df['area_hectares']
    merged_df['Grazing_units_relative_bulk'] = merged_df['Grazing_units_absolute_bulk'] / merged_df['area_hectares']
    merged_df['Browsing_units_relative_bulk'] = merged_df['Browsing_units_absolute_bulk'] / merged_df['area_hectares']

    merged_df['Browsing_units/ha_woody_vegetation_selective'] = merged_df['Browsing_units_absolute_selective'] / merged_df['area_hectares'] / merged_df['mean_woody_fraction']
    merged_df['Grazing_units/ha_herbaceous_vegetation_selective'] = merged_df['Grazing_units_absolute_selective'] / merged_df['area_hectares'] / merged_df['mean_herbaceous_fraction']
    merged_df['Browsing_units/ha_woody_vegetation_bulk'] = merged_df['Browsing_units_absolute_bulk'] / merged_df['area_hectares'] / merged_df['mean_woody_fraction']
    merged_df['Grazing_units/ha_herbaceous_vegetation_bulk'] = merged_df['Grazing_units_absolute_bulk'] / merged_df['area_hectares'] / merged_df['mean_herbaceous_fraction']

    # Save updated DataFrame
    output_file = f"{output_base_path}/results.csv"
    merged_df.to_csv(output_file, index=False)
    progress_label2.config(text=f"Feeding units and vegetation categories results table saved to: {output_file}")

    # Update the second plot canvas
    relative_feeding_units_plot()

### 10. Download GEE script ____________________________________________________________________________________________________
def download_text_file(): # Function to download GEE script. Long term aim: include the retrival in the app using the GEE python API. For example through openeo https://identity.dataspace.copernicus.eu/auth/realms/CDSE/device?user_code=ASGI-RHVG
    content = """
    //Go to https://code.earthengine.google.com/

    // Define the area of interest using a shapefile
    var geometry = ee.FeatureCollection('path/to/your/uploaded/shapefile');

    // Filter Sentinel-2 imagery for the specified date range and cloud cover
    var sentinel2 = ee.ImageCollection('COPERNICUS/S2')
      .filterBounds(geometry)
      .filterDate('YYYY-MM-DD', 'YYYY-MM-DD')
      .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 1);

    // Select the latest image
    var image = sentinel2.limit(1, 'system:time_start', false).first();

    // Select a median image
    //var image = sentinel2.median(); // if the image is incomplete, substitute the above line with this line

    image = image.toFloat().resample('bilinear').reproject(image.select('B2').projection());

    // Clip the image to the specified geometry
    var clippedImage = image.clip(geometry);
    // if the above is not working try: var clippedImage = image.clip(geometry.geometry());

    // Select the bands of interest
    var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12'];

    // Select the desired bands from the clipped image
    var selectedBands = clippedImage.select(bands);

    // Print the clipped image to the map
    // Map.addLayer(clippedImage, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'Clipped Image');

    // Center the map on the specified geometry
    Map.centerObject(geometry, 10);

    // Define export parameters
    var exportParams = {
      image: selectedBands,
      description: 'description', // Change the description as needed
      scale: 10,
      folder: 'GEE_exports' // Specify the folder in your Google Drive
    };

    // Export the image to Google Drive
    Export.image.toDrive(exportParams);
    """
    
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'w') as file:
            file.write(content)
        messagebox.showinfo("Success", f"File saved to: {file_path}")

### 11. Graphical user interface: Setup and style ____________________________________________________________________________________________________
# Base color scheme
DARK_BACKGROUND = "#111111"
LIGHT_TEXT = "#FFFFFF"
DARK_TEXT = "#111111"
DARK_BUTTON = "#555555"
DARK_ENTRY = "#FFFFFF"
HOVER_BG = "#666666"

# Main window
app = tk.Tk()
app.title("Herbivory Monitoring Tool")
app.configure(bg=DARK_BACKGROUND)

# Style for ttk elements
style = ttk.Style()
style.theme_use('clam') # or 'alt', 'default', 'classic', 'vista', 'clam'

# Font
default_font = ('Arial', 10)
big_font = ('Arial', 12)

# General style configuration
style.configure('TLabel', background=DARK_BACKGROUND, foreground=LIGHT_TEXT, font=default_font)
style.configure('TButton', background=DARK_BUTTON, foreground=LIGHT_TEXT, font=default_font)
style.configure('TEntry', background=DARK_ENTRY, foreground=DARK_TEXT, font=default_font)
style.configure('TFrame', background=DARK_BACKGROUND)
style.map('TButton', background=[('active', HOVER_BG)])

# Notebook style
style.configure('TNotebook', background=DARK_BACKGROUND, borderwidth=0)
style.configure('TNotebook.Tab', background=DARK_BACKGROUND, foreground=LIGHT_TEXT, lightcolor=DARK_BUTTON, padding=[5, 5])
style.map('TNotebook.Tab', background=[('selected', DARK_BUTTON), ('active', HOVER_BG)])

# Treeview style
style.configure('Treeview', background="#383838", foreground=LIGHT_TEXT, fieldbackground="#383838")
style.map('Treeview', background=[('selected', HOVER_BG)])

# Creating notebook
notebook = ttk.Notebook(app)
notebook.pack(fill='both', expand=True, padx=2, pady=3)

# Tabs
images_frame = ttk.Frame(notebook, style='TFrame')
combined_operations_frame = ttk.Frame(notebook, style='TFrame')

# Add frames to the Notebook with respective titles
notebook.add(images_frame, text='Herbivore Abundance')
notebook.add(combined_operations_frame, text='Vegetation Availability')

### 12. Tab 1: Camera trap images ____________________________________________________________________________________________________
ttk.Button(images_frame, text="Select general output directory", command=browse_and_create_output_directory).grid(row=0, column=0, columnspan=1, padx=10, pady=5)

images_folder_entry = ttk.Entry(images_frame, width=40, font=default_font)
images_folder_entry.grid(row=0, column=2, padx=10, pady=5)

images_folder_button = ttk.Button(images_frame, text="Navigate to image folder", command=lambda: browse_folder(images_folder_entry))
images_folder_button.grid(row=0, column=1, padx=10, pady=5)

perform_yolo_button = ttk.Button(images_frame, text="Classify images", command=perform_yolo_predictions_threaded)
perform_yolo_button.grid(row=1, column=1, padx=10, pady=5)

progress_label1 = ttk.Label(images_frame, text="", font=default_font)
progress_label1.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

thresh_frame = ttk.Frame(images_frame)
thresh_frame.grid(row=1, column=0, padx=10, pady=5, sticky='w')

thresh_label = ttk.Label(thresh_frame, text="Probability threshold:")
thresh_label.pack(side=tk.LEFT, padx=5)

prob_thresh_entry = ttk.Entry(thresh_frame, width=10)
prob_thresh_entry.pack(side=tk.LEFT)
prob_thresh_entry.insert(0, "0.75")  # Default value of 0.75

detection_table = ttk.Treeview(images_frame, columns=('Species', 'Count', 'Select'), show='headings', height=6)
detection_table.grid(row=4, column=0, columnspan=1, padx=(10, 5), pady=10, sticky='nsew')

detection_table.heading('Species', text='Species')
detection_table.heading('Count', text='Count')
detection_table.heading('Select', text='Select')

detection_table.column('Species', width=30, minwidth=30, stretch=tk.YES)
detection_table.column('Count', width=30, minwidth=30, stretch=tk.YES)
detection_table.column('Select', width=30, minwidth=30, stretch=tk.YES)

detection_table.bind('<Button-1>', toggle_checkbox)

detection_table.tag_configure('checkbox', image='')

images_frame.columnconfigure(0, weight=1)
images_frame.columnconfigure(1, weight=1)
images_frame.columnconfigure(2, weight=1)
images_frame.rowconfigure(4, weight=1)

organize_images_button = ttk.Button(images_frame, text="Copy images to folders for selected species", command=organize_images)
organize_images_button.grid(row=1, column=2, columnspan=2, padx=10, pady=10)

canvas_frame = ttk.Frame(images_frame, style='TFrame')
canvas_frame.grid(row=4, column=1, columnspan=2, padx=(5, 10), pady=10, sticky='nsew')

figure_canvas = FigureCanvasTkAgg(plt.Figure(figsize=(6, 4)), master=canvas_frame)
figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

perform_RAI_button = ttk.Button(images_frame, text="Perform RAI analysis", command=lambda: RAI(figure_canvas))
perform_RAI_button.grid(row=2, column=2, columnspan=1, padx=10, pady=5)

### 13. Tab 2: Vegetation categories ____________________________________________________________________________________________________
input_raster_label = ttk.Label(combined_operations_frame, text="Sentinel-2 input raster:")
input_raster_label.grid(row=0, column=0, padx=10, pady=5)

input_raster_entry = ttk.Entry(combined_operations_frame, width=50)
input_raster_entry.grid(row=0, column=1, padx=10, pady=5)

input_raster_button = ttk.Button(combined_operations_frame, text="Browse", command=lambda: browse_file_raster(input_raster_entry))
input_raster_button.grid(row=0, column=2, padx=10, pady=5)

run_gf_button = ttk.Button(combined_operations_frame, text="Run vegetation category model", command=lambda: run_vegetation_category_model())
run_gf_button.grid(row=1, column=3, columnspan=1, padx=10, pady=5)

download_button = ttk.Button(combined_operations_frame, text="Download GEE script", command=download_text_file)
download_button.grid(row=0, column=3,columnspan=1, padx=10, pady=5)

polygon_shp_label = ttk.Label(combined_operations_frame, text="Shapefile of the protected area:", font=default_font)
polygon_shp_label.grid(row=1, column=0, padx=10, pady=5)

polygon_shp_entry = ttk.Entry(combined_operations_frame, width=50)
polygon_shp_entry.grid(row=1, column=1, padx=10, pady=5)

polygon_shp_button = ttk.Button(combined_operations_frame, text="Browse", command=lambda: browse_file_shp(polygon_shp_entry))
polygon_shp_button.grid(row=1, column=2, padx=10, pady=5)

points_csv_label = ttk.Label(combined_operations_frame, text="CSV file with waterpoint coordinates:", font=default_font)
points_csv_label.grid(row=2, column=0, padx=10, pady=5)

points_csv_entry = ttk.Entry(combined_operations_frame, width=50)
points_csv_entry.grid(row=2, column=1, padx=10, pady=5)

points_csv_button = ttk.Button(combined_operations_frame, text="Browse", command=lambda: browse_file_csv(points_csv_entry))
points_csv_button.grid(row=2, column=2, padx=10, pady=5)

run_voronoi_button = ttk.Button(combined_operations_frame, text="Run analysis", command=run_voronoi_button_callback)
run_voronoi_button.grid(row=2, column=3, columnspan=1, padx=10, pady=5)

progress_label2 = ttk.Label(combined_operations_frame, text="", font=default_font)
progress_label2.grid(row=3, column=0, columnspan=3, padx=10, pady=5)

combined_canvas_frame = ttk.Frame(combined_operations_frame, style='TFrame')
combined_canvas_frame.grid(row=4, column=0, columnspan=4, padx=(10, 10), pady=10, sticky='nsew')

class_raster_voronoi_canvas = FigureCanvasTkAgg(plt.Figure(figsize=(6, 4)), master=combined_canvas_frame)
class_raster_voronoi_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

feeding_units_canvas = FigureCanvasTkAgg(plt.Figure(figsize=(6, 4)), master=combined_canvas_frame) 
feeding_units_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

app.mainloop()