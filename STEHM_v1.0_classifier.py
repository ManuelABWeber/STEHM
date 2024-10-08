########################################################################################################################################################################################
### Spatially and Temporally Explicit Herbivory Monitoring in Water-limited Savannas
### Camera Trap Imagery Classifier
### Manuel Weber
### https://github.com/ManuelABWeber/STEHM.git
########################################################################################################################################################################################
# Rendering command: pyinstaller --noconfirm --onefile --console --icon "C:\Giraffe.ico" --add-data "C:\yoloweightsv10_lib.pt;." --add-data "C:\python_venv\Lib\site-packages\ultralytics;ultralytics/" --add-data "C:\Giraffe.ico;."  "C:\STEHM_v1.0_classifier.py"

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
import pandas as pd
import os
import sys
from PIL import Image, ExifTags
from datetime import date
import geopandas as gpd
from tkinter.filedialog import asksaveasfilename
import shutil

# Handle the base path for PyInstaller
try:
    base_path = sys._MEIPASS  # Path where PyInstaller unpacks files
except AttributeError:
    base_path = os.path.abspath(".")  # If not bundled, use the current directory

output_base_path = ""
results_df = pd.DataFrame()
checkbox_vars = {}

def check_and_remove_corrupted_files(directory_path):
        print("check starting")
        count = 0
        for root, dirs, files in os.walk(directory_path):
            print(directory_path)
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                    except (IOError, SyntaxError) as e:
                        print(f"Corrupted file detected and removed: {file_path}")
                        os.remove(file_path)
                        count += 1
        print(f"Total corrupted files removed: {count}")

class HerbivoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("STEHM v1.0 Classifier")
        self.root.iconbitmap(os.path.join(base_path, "Giraffe.ico"))
        self.checkbox_vars = {}  # Initialize checkbox_vars here
        self.setup_ui()

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Button(frame, text="Select image folder", command=self.browse_folder).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Classify images", command=self.classify_images).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(frame, text="Copy images to folders for selected species", command=self.organize_images).grid(row=0, column=3, padx=5, pady=5)

        conf_frame = ttk.Frame(frame)
        conf_frame.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(conf_frame, text="Confidence threshold:").pack(side=tk.LEFT)
        self.prob_thresh_entry = ttk.Entry(conf_frame, width=10)
        self.prob_thresh_entry.pack(side=tk.LEFT)
        self.prob_thresh_entry.insert(0, "0.75")

        # Table to display species counts and checkboxes
        self.species_table_frame = ttk.Frame(frame)
        self.species_table_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=10)

        # Set up table headers
        ttk.Label(self.species_table_frame, text="Species").grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(self.species_table_frame, text="Image Count").grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(self.species_table_frame, text="Select").grid(row=0, column=2, padx=5, pady=5)

    def browse_folder(self):
        messagebox.showinfo("Information", "Please provide the path to a directory with folders that correspond to waterpoints. Only images in waterpoint folders will be classified.")
        self.folder_path = filedialog.askdirectory()
        if not self.folder_path:
            messagebox.showerror("Error", "No folder selected.")
        else:
            print(f"Selected folder: {self.folder_path}\n")

    def get_image_metadata(self, image_path):
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()
            if not exif_data:
                return None
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == "DateTimeOriginal":
                    return value
        except Exception as e:
            print(f"Error extracting metadata for {image_path}: {e}\n")
        return None
    
    

    def classify_images(self):
        if not hasattr(self, 'folder_path') or not self.folder_path:
            messagebox.showerror("Error", "No folder selected.")
            return
        
        check_and_remove_corrupted_files(self.folder_path)
        print("Corrupted file check completed.\n")
        model_path = os.path.join(base_path, "yoloweightsv10_lib.pt")
        
        try:
            yolo_model = YOLO(model_path)
            print(f"YOLO model loaded successfully\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
            return

        try:
            threshold = float(self.prob_thresh_entry.get())
            print("Classifying images...\n")
            image_path = self.folder_path
            results = yolo_model.predict(source=os.path.join(image_path, '*/*'), stream=True, save=False, conf=threshold)

            temp_results = []
            for r in results:
                max_confidences = {f"{species}_confidence": 0 for species in yolo_model.names.values()}
                individual_counts = {species: 0 for species in yolo_model.names.values()}
                date_time = self.get_image_metadata(r.path)
                if date_time is None:
                    date_time = "NA"
                    print(f"Metadata not found for image: {r.path}\n")

                for idx, class_id in enumerate(r.boxes.cls.numpy()):
                    species = yolo_model.names[int(class_id)]
                    confidence = r.boxes.conf.numpy()[idx]
                    individual_counts[species] += 1  # Count the number of individuals detected
                    if confidence > max_confidences[f"{species}_confidence"]:
                        max_confidences[f"{species}_confidence"] = confidence

                max_confidences['Name'] = r.path
                max_confidences['Date_Time'] = date_time
                max_confidences.update({f"{species}_count": count for species, count in individual_counts.items()})
                temp_results.append(max_confidences)

            species_columns = [f"{species}_confidence" for species in yolo_model.names.values()]
            count_columns = [f"{species}_count" for species in yolo_model.names.values()]
            columns = ['Name', 'Date_Time'] + species_columns + count_columns

            global results_df
            results_df = pd.DataFrame(temp_results, columns=columns)

            print("All images classified.\n")

            # Let the user choose where to save the file, with a default name
            csv_file = asksaveasfilename(defaultextension=".csv",
                                        initialfile="full_image_annotations.csv",
                                        filetypes=[("CSV files", "*.csv")],
                                        title="Save the classified results")

            # Check if the user canceled the save dialog
            if csv_file:
                results_df.to_csv(csv_file, index=False)
                print(f"Success. Predictions saved under {csv_file}.\n")
            else:
                messagebox.showwarning("Warning", "Save operation canceled. Results not saved.")

            # Update species table with new data
            self.update_species_table()

            na_count = results_df['Date_Time'].isna().sum()
            total_count = len(results_df)
            messagebox.showinfo("Success", f"Predictions saved. {na_count} out of {total_count} images lack metadata and cannot be used to compute RAIs.")
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {e}")

    def organize_images(self):
        global results_df

        if results_df.empty:
            messagebox.showerror("Error", "No classification results found. Perform classification first.")
            return

        # Prompt the user to select the base directory where folders will be created
        output_base_path = filedialog.askdirectory(title="Select Output Directory")
        if not output_base_path:
            messagebox.showerror("Error", "No directory selected.")
            return

        threshold = float(self.prob_thresh_entry.get()) if self.prob_thresh_entry.get() else 0.75
        selected_species = [species for species, var in self.checkbox_vars.items() if var.get()]

        if not selected_species:
            messagebox.showerror("Error", "No species selected.")
            return

        for species in selected_species:
            species_name = species.split('_', 1)[-1]
            species_confidence = f"{species_name}_confidence"

            filtered_df = results_df[results_df[species_confidence] >= threshold]

            for _, row in filtered_df.iterrows():
                image_path = row['Name']
                # Extract the site name as the set of characters between the last two "/" or "\"
                site_name = os.path.normpath(image_path).split(os.sep)[-2]

                site_folder = os.path.join(output_base_path, species_name, site_name)  # Create subfolder for each site

                if not os.path.exists(site_folder):
                    os.makedirs(site_folder)

                try:
                    shutil.copy(image_path, site_folder)
                except Exception as e:
                    self.progress_text.insert(tk.END, f"Error moving image {image_path}: {str(e)}\n")

        messagebox.showinfo("Success", "Images have been organized into species and site folders.")

    def update_species_table(self):
        # Clear existing table entries
        for widget in self.species_table_frame.winfo_children()[3:]:  # Skip the headers
            widget.destroy()

        # Check if results_df is populated
        if results_df.empty:
            return

        # Populate the table with species counts and checkboxes
        species_names = [col.replace("_count", "") for col in results_df.columns if col.endswith("_count")]
        for i, species in enumerate(species_names):
            # Count the number of images where the species was detected (i.e., count > 0)
            image_count = (results_df[f"{species}_count"] > 0).sum()

            # Species name label
            ttk.Label(self.species_table_frame, text=species.capitalize()).grid(row=i+1, column=0, padx=5, pady=5)

            # Image count label
            ttk.Label(self.species_table_frame, text=str(image_count)).grid(row=i+1, column=1, padx=5, pady=5)

            # Checkbox for selecting species
            var = tk.BooleanVar(value=False)
            self.checkbox_vars[species] = var
            ttk.Checkbutton(self.species_table_frame, variable=var).grid(row=i+1, column=2, padx=5, pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    root.iconbitmap(os.path.join(base_path, "Giraffe.ico"))
    app = HerbivoryApp(root)
    root.mainloop()