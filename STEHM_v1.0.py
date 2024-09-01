########################################################################################################################################################################################
### Spatially and Temporally Explicit Herbivory Monitoring in Water-limited Savannas
### Desktop Application
### Manuel Weber
### https://github.com/ManuelABWeber/STEHM.git
########################################################################################################################################################################################
# rendering command: pyinstaller --noconfirm --onefile --console --icon "C:\Giraffe.ico" --add-data "C:\Giraffe.ico;." --add-data "C:\DLmodel.keras;." --add-data "C:\python_venv\Lib\site-packages\bqplot;bqplot/" --add-data "C:\python_venv\Lib\site-packages\geemap;geemap/" --hidden-import "rasterio.vrt" --hidden-import "rasterio.sample"  "C:\STEHM_v1.0.py"

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Menu
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk as NavigationToolbar2
import os
import subprocess
import shutil
import glob
import sys
import ee
import geemap
from datetime import date, timedelta, datetime
import numpy as np
import matplotlib.colors as mcolors
import rasterio
from PIL import Image, ExifTags
from rasterio.mask import geometry_mask, mask
from rasterio.merge import merge
import shapely
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, mapping, box
from matplotlib.patches import Patch
from tensorflow.keras.models import load_model


output_base_path = ""
results_df = pd.DataFrame()
checkbox_vars = {}

try:
    script_dir = sys._MEIPASS  # Path where PyInstaller unpacks files
except AttributeError:
    script_dir = os.path.abspath(".")  # If not bundled, use the current directory

class HerbivoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("STEHM v1.0")
        self.root.iconbitmap(os.path.join(script_dir, "Giraffe.ico"))
        self.site_names = []
        self.plot_vars = {}  # Dictionary to hold variables for checkboxes
        self.checkbox_vars = {}  # Dictionary to hold species checkbox variables
        self.setup_ui()
        

    def setup_ui(self):
        # Main layout frames
        control_frame = ttk.Frame(self.root, width=500)  # Set a fixed width for the control frame
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        visualization_frame = ttk.Frame(self.root)
        visualization_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Step 1: Initialization frame
        init_frame = ttk.Labelframe(control_frame, text="Step 1: Initialization", padding=10)
        init_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5, columnspan=3)

        ttk.Button(init_frame, text="Select output directory", command=self.browse_and_create_output_directory).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(init_frame, text="Import boundaries", command=self.import_boundaries).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(init_frame, text="Import waterpoints", command=self.load_waterpoints_csv).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(init_frame, text ="Reset all", command=self.reset_all).grid(row=0, column = 3, padx=5, pady=5)

        menu_button = ttk.Menubutton(init_frame, text="Export", direction='below')
        menu = Menu(menu_button, tearoff=False)
        menu.add_command(label="Download user guide", command=self.download_user_guide)
        menu.add_command(label="Download waterpoint template", command=self.download_waterpoint_template)
        menu.add_command(label="Download annotations template", command=self.download_annotation_template)
        menu.add_command(label="Eport plot", command=self.export_plot)
        menu_button["menu"] = menu
        menu_button.grid(row=0, column=4, sticky="w")

        # Step 2: Camera trap images frame
        cam_trap_frame = ttk.Labelframe(control_frame, text="Step 2: Camera trap images", padding=10)
        cam_trap_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5, columnspan=3)

        # Combine label and entry into one cell
        conf_frame = ttk.Frame(cam_trap_frame)
        conf_frame.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(conf_frame, text="Confidence threshold:").pack(side=tk.LEFT)
        self.prob_thresh_entry = ttk.Entry(conf_frame, width=5)
        self.prob_thresh_entry.pack(side=tk.LEFT)
        self.prob_thresh_entry.insert(0, "0.75")  # Default value of 0.75

        #ttk.Button(cam_trap_frame, text="Classify images", command=self.classify_images).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(cam_trap_frame, text="Compute RAIs", command=self.compute_rais).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(cam_trap_frame, text="Copy images to folders for selected species", command=self.organize_images).grid(row=0, column=2, padx=5, pady=5)

        # Step 3: Vegetation frame
        veg_frame = ttk.Labelframe(control_frame, text="Step 3: Vegetation", padding=10)
        veg_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5, columnspan=3)

        month_frame = ttk.Frame(veg_frame)
        month_frame.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(month_frame, text="Month [MM]:").pack(side=tk.LEFT)
        self.month_entry = ttk.Entry(month_frame, width=5)
        self.month_entry.pack(side=tk.LEFT)
        self.month_entry.insert(0, "01")  # Default value of 0.75

        year_frame = ttk.Frame(veg_frame)
        year_frame.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(year_frame, text="Year [YYYY]:").pack(side=tk.LEFT)
        self.year_entry = ttk.Entry(year_frame, width=5)
        self.year_entry.pack(side=tk.LEFT)
        self.year_entry.insert(0, "2024")

        cloud_frame = ttk.Frame(veg_frame)
        cloud_frame.grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(cloud_frame, text="Cloud cover [%]:").pack(side=tk.LEFT)
        self.cloud_entry = ttk.Entry(cloud_frame, width=5)
        self.cloud_entry.pack(side=tk.LEFT)
        self.cloud_entry.insert(0, "1") 
        
        ttk.Button(veg_frame, text="Acquire image", command=self.acquire_sentinel).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(veg_frame, text="Run model", command=self.load_and_run_vegetation_model).grid(row=0, column=4, columnspan=1, padx=5, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=visualization_frame)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(visualization_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2(self.fig_canvas, toolbar_frame)
        toolbar.update()


        # Data display frame with scrolling
        self.canvas_frame = ttk.Frame(control_frame)
        self.canvas_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

        # Create a canvas
        self.canvas = tk.Canvas(self.canvas_frame)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Add a scrollbar for vertical scrolling
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")

        # Add a scrollbar for horizontal scrolling
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        # Create a frame inside the canvas
        self.data_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.data_frame, anchor="nw")

        # Make sure the canvas is updated to match the size of the data frame
        self.data_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Configure the canvas frame to expand with the window
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        # Bind mouse scroll events to scroll the canvas

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def reset_all(self):
        # Confirm with the user before resetting
        confirm_reset = messagebox.askyesno("Confirm Reset", "Are you sure you want to reset the application to its initial state? This will clear all data and settings.")
        
        if not confirm_reset:
            return  # If the user chooses "No", exit the function

        # Clear all global variables
        global output_base_path, results_df, checkbox_vars
        output_base_path = ""
        results_df = pd.DataFrame()
        checkbox_vars = {}
        
        # Clear site names and plot variables
        self.site_names = []
        self.plot_vars = {}
        self.checkbox_vars = {}
        
        # Reset text entries to their default values
        self.prob_thresh_entry.delete(0, tk.END)
        self.prob_thresh_entry.insert(0, "0.75")
        
        self.month_entry.delete(0, tk.END)
        self.month_entry.insert(0, "01")
        
        self.year_entry.delete(0, tk.END)
        self.year_entry.insert(0, str(datetime.now().year))
        
        self.cloud_entry.delete(0, tk.END)
        self.cloud_entry.insert(0, "1")
        
        # Clear the canvas and data frame
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        
        self.ax.clear()
        self.fig_canvas.draw()
        
        # Clear any loaded data
        if hasattr(self, 'boundaries_gdf'):
            delattr(self, 'boundaries_gdf')
        
        if hasattr(self, 'waterpoints_df'):
            delattr(self, 'waterpoints_df')
        
        if hasattr(self, 'waterpoints_gdf'):
            delattr(self, 'waterpoints_gdf')
        
        if hasattr(self, 'voronoi_polygons_gdf'):
            delattr(self, 'voronoi_polygons_gdf')
        
        if hasattr(self, 'vegetation_raster'):
            delattr(self, 'vegetation_raster')
        
        if hasattr(self, 'vegetation_raster_array'):
            delattr(self, 'vegetation_raster_array')
        

        # Reset the working directory
        os.chdir(script_dir)  # Reset to the script's directory or initial working directory

        # Notify the user that the reset is complete
        messagebox.showinfo("Reset Complete", "The application has been reset to its initial state.")


    def browse_and_create_output_directory(self):
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

    def import_boundaries(self):
        file_path = filedialog.askopenfilename(filetypes=[("Shapefiles", "*.shp"), ("KML files", "*.kml")])
        if file_path:
            if file_path.endswith(".shp"):
                self.boundaries_gdf = gpd.read_file(file_path)
            elif file_path.endswith(".kml"):
                self.boundaries_gdf = gpd.read_file(file_path, driver='KML')

            # Plot the boundaries outline as a black line
            self.ax.clear()
            self.boundaries_gdf.boundary.plot(ax=self.ax, edgecolor='black')

            self.fig_canvas.draw()

    def load_waterpoints_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.waterpoints_df = pd.read_csv(file_path)
            self.waterpoints_df['Site'] = self.waterpoints_df['Site'].astype(str)  # Ensure Site is treated as string
            self.site_names = self.waterpoints_df['Site'].tolist()

            if hasattr(self, 'site_names') and self.site_names:
                self.waterpoints_gdf = gpd.GeoDataFrame(
                    self.waterpoints_df, 
                    geometry=gpd.points_from_xy(self.waterpoints_df.Longitude, self.waterpoints_df.Latitude)
                )
                # Ensure the waterpoints GeoDataFrame has a CRS
                if self.waterpoints_gdf.crs is None:
                    self.waterpoints_gdf.set_crs(epsg=4326, inplace=True)

                self.waterpoints_gdf.plot(ax=self.ax, color='black', markersize=5)
                for x, y, label in zip(self.waterpoints_gdf.geometry.x, self.waterpoints_gdf.geometry.y, self.site_names):
                    self.ax.text(x, y, label, fontsize=8, color='black')

                # Voronoi tessellation
                self.voronoi_polygons_gdf = self.create_voronoi_polygons_old(self.waterpoints_gdf)

                self.fig_canvas.draw()

                # Update data display with area information
                self.update_data_display(["Sites"] + self.site_names)
                self.update_voronoi_area_row()  # Ensure "Area (ha)" row appears immediately

                print(f"Loaded waterpoints: {', '.join(self.site_names)}\n")

    def create_voronoi_polygons_old(self, waterpoints_gdf):
        if waterpoints_gdf.crs is None:
            waterpoints_gdf.set_crs(epsg=4326, inplace=True)

        if self.boundaries_gdf.crs is None:
            self.boundaries_gdf.set_crs(epsg=4326, inplace=True)

        coords = np.array(waterpoints_gdf[['Longitude', 'Latitude']])
        polygon = self.boundaries_gdf.geometry.iloc[0]

        bound = polygon.buffer(100).envelope.boundary

        boundarypoints = [bound.interpolate(distance=d) for d in range(0, int(np.ceil(bound.length)), 100)]
        boundarycoords = np.array([[p.x, p.y] for p in boundarypoints])

        all_coords = np.concatenate((boundarycoords, coords))

        vor = Voronoi(points=all_coords)

        lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]

        polys = shapely.ops.polygonize(lines)

        voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys), crs=waterpoints_gdf.crs)

        if voronois.crs != self.boundaries_gdf.crs:
            voronois = voronois.to_crs(self.boundaries_gdf.crs)

        voronois = gpd.clip(voronois, self.boundaries_gdf)

        # Assign site names to the polygons
        coords = waterpoints_gdf.geometry.apply(lambda p: (p.x, p.y)).tolist()
        site_names = []
        for poly in voronois.geometry:
            centroid = poly.centroid
            nearest_site = min(coords, key=lambda p: centroid.distance(shapely.geometry.Point(p)))
            site_name = waterpoints_gdf[waterpoints_gdf.geometry == shapely.geometry.Point(nearest_site)]['Site'].values[0]
            site_names.append(site_name)
        voronois['Site'] = site_names

        voronois_3857 = voronois.to_crs(epsg=3857)

        voronois['Area (ha)'] = voronois_3857.geometry.area / 10000

        output_voronoi_path = os.path.join(output_base_path, 'voronoi_polygons.shp')
        voronois.to_file(output_voronoi_path)

        return voronois

    def update_voronoi_area_row(self):
        if hasattr(self, 'voronoi_polygons_gdf'):
            areas = self.voronoi_polygons_gdf['Area (ha)'].tolist()
            row_data = ['Area [ha]'] + [f"{area:.2f}" for area in areas]
            row_index = 1  # Assuming this is the first row after the header
            
            # Remove any existing checkboxes in this row to prevent duplicates
            for widget in self.data_frame.grid_slaves(row=row_index):
                widget.destroy()
            
            # Add the row to the table
            self.add_data_row(row_data, row_index)
            
            # Adjust column widths: First column for checkbox, second column wider, others narrower
            for col_index, data in enumerate(row_data):
                if col_index == 0:
                    width = 25  # Narrow width for the checkbox column
                else:
                    width = 7  # Narrower width for the other columns
                label = ttk.Label(self.data_frame, text=data, borderwidth=1, relief="solid", width=width)
                label.grid(row=row_index, column=col_index + 1, sticky='nsew')  # Start from column 1

            # Add the checkbox only if it doesn't already exist
            if 'Area [ha]' not in self.plot_vars:
                plot_var = tk.BooleanVar(value=True)
                self.plot_vars['Area [ha]'] = plot_var
                plot_checkbox = ttk.Checkbutton(self.data_frame, variable=plot_var, command=self.update_voronoi_plot)
                plot_checkbox.grid(row=row_index, column=len(self.site_names) + 1, sticky='nsew')

    def clip_raster_to_polygons(self, band):
        masked_band = np.zeros_like(band, dtype=float)
        for idx, row in self.voronoi_polygons_gdf.iterrows():
            mask = geometry_mask([row.geometry], transform=self.vegetation_raster.transform, invert=True, out_shape=band.shape)
            masked_band[mask] = band[mask]
        return masked_band

    def get_raster_extent(self):
        transform = self.vegetation_raster.transform
        width = self.vegetation_raster.width
        height = self.vegetation_raster.height
        return (transform[2], transform[2] + width * transform[0], transform[5] + height * transform[4], transform[5])

    def get_polygon_color(self, site, class_name):
        df = None
        if hasattr(self, 'rai_df') and class_name in self.rai_df.columns:
            df = self.rai_df
        elif hasattr(self, 'relative_feeding_units_df') and class_name in self.relative_feeding_units_df.columns:
            df = self.relative_feeding_units_df

        if df is not None:
            site_data = df[df['Waterpoint'] == site]
            if not site_data.empty:
                max_value = df[class_name].max()
                value = site_data[class_name].values[0]
                if not pd.isna(value) and max_value > 0:  # Ensure value and max_value are valid
                    normalized_value = value / max_value
                    rgba_color = plt.cm.Reds(normalized_value)
                    return rgba_color[:3] + (0.7,)
        return None  # No color if no values

    def update_data_display(self, columns):
        # Clear previous data
        for widget in self.data_frame.winfo_children():
            widget.grid_forget()  # Clears the grid without destroying the widget
            widget.destroy()       # Destroys the widget

        # Add the header for the "Plot" column (checkboxes) first
        plot_label = ttk.Label(self.data_frame, text="Plot", borderwidth=1, relief="solid", width=7)
        plot_label.grid(row=0, column=0, sticky='nsew')

        # Add the rest of the headers starting from column 1
        for col_index, col in enumerate(columns):
            if col_index == 0:
                width = 25  # "Sites" column should be wider
            else:
                width = 7  # All other columns, including the first site column, should be narrower
            label = ttk.Label(self.data_frame, text=col, borderwidth=1, relief="solid", width=width)
            label.grid(row=0, column=col_index + 1, sticky='nsew')  # Start from column 1

    def add_data_row(self, row_data, row_index):
        # Add the checkbox in the leftmost column
        class_name = row_data[0]
        self.add_class_checkbox(class_name, row_index)  # Add checkbox in column 0

        # Add the rest of the row data starting from column 1
        for col_index, data in enumerate(row_data):
            if col_index == 0:
                width = 25  # Narrow width for the checkbox column
            else:
                width = 7  # Narrower width for the site columns
            label = ttk.Label(self.data_frame, text=data, borderwidth=1, relief="solid", width=width)
            label.grid(row=row_index, column=col_index + 1, sticky='nsew')  # Start from column 1

    def update_plot(self):
        self.update_voronoi_plot_based_on_checkboxes()

    def browse_file(self, entry_field, message, file_types):
        messagebox.showinfo("Information", message)
        filepath = filedialog.askopenfilename(filetypes=file_types)
        if filepath:
            entry_field.delete(0, tk.END)
            entry_field.insert(0, filepath)
        else:
            messagebox.showerror("Error", "No file selected.")

    def browse_file_raster(self):
        filepath = filedialog.askopenfilename(filetypes=[("TIF files", "*.tif")])
        if filepath:
            return filepath
        else:
            messagebox.showerror("Error", "No file selected.")
            return None

    def compute_relative_feeding_units(self):
        # Call the output function to ensure calculations and CSV saving are done once
        self.output()

        # Load the results CSV to update the DataFrame
        relative_feeding_units_file = f"{output_base_path}/results.csv"
        self.relative_feeding_units_df = pd.read_csv(relative_feeding_units_file)

        print(f"Relative feeding units loaded from: {relative_feeding_units_file}\n")

    def save_feeding_units_plot(self):
        sorted_results = self.relative_feeding_units_df.sort_values(by='Waterpoint')

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
        ax.set_yticklabels(sorted_results['Waterpoint'])
        ax.set_xlabel('Feeding units per camera trap day and ha of woody or herbaceous vegetation')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{abs(x):.2f}"))
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
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize='small', handlelength=1, handleheight=1, markerscale=0.8)

        plot_path = os.path.join(output_base_path, 'relative_feeding_units.png')
        plt.savefig(plot_path)
        plt.close(fig)

    def load_and_run_vegetation_model(self):
        global output_base_path
        output_raster_path = f"{output_base_path}/vegetation_categories_raster.tif"
        sentinel_raster_path = f"{output_base_path}/sentinel_2_image.tif"

        # Check if the classification is already complete
        if os.path.exists(output_raster_path):
            messagebox.showinfo("Information", "Vegetation categories raster already exists. Loading the existing raster.")
            self.load_existing_vegetation_raster(output_raster_path)
        else:
            # Check if the Sentinel-2 raster exists
            if os.path.exists(sentinel_raster_path):
                messagebox.showinfo("Information", f"Found Sentinel-2 raster at {sentinel_raster_path}. Running model on this raster.")
                self.classify_raster(sentinel_raster_path) 
            else:
                # Prompt the user to browse for a different raster if the default one doesn't exist
                messagebox.showwarning("Warning", f"No Sentinel-2 raster found at {sentinel_raster_path}. Please select a raster.")
                selected_raster_path = self.browse_file_raster()
                
                # Check if a valid raster file path was selected
                if selected_raster_path:  # Assuming browse_file_raster() returns the selected file path
                    self.run_vegetation_category_model(selected_raster_path)
                else:
                    messagebox.showerror("Error", "No raster file selected. Cannot run the vegetation model.")


        # Compute relative feeding units and add them to the table
        self.compute_relative_feeding_units()
        self.save_feeding_units_plot()

        # Ensure the table is updated only once after all calculations are done
        self.add_relative_feeding_units_to_table()

    def load_existing_vegetation_raster(self, raster_path):
        try:
            self.vegetation_raster = rasterio.open(raster_path)
            self.vegetation_raster_array = self.vegetation_raster.read()
            print(f"Vegetation category raster loaded from: {raster_path}\n")
            self.update_data_table_with_vegetation()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the vegetation raster: {e}")

    def run_vegetation_category_model(self, input_raster_path):
        if not os.path.exists(input_raster_path):
            messagebox.showerror("Error", "Input raster file not found.")
            return

        self.classify_raster(input_raster_path)

    def classify_raster(self, input_raster_path):
        global output_base_path
        if not output_base_path:
            messagebox.showerror("Error", "Output directory not set.")
            return

        print("Loading model...\n")

        model_path = os.path.join(script_dir, "DLmodel.keras")

        try:
            dl_model = load_model(model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the model: {e}")
            return

        raster_array = self.raster_to_array(input_raster_path)
        if raster_array is None:
            return

        print("Raster normalization...\n")

        # Normalize by dividing the bands by 10,000
        raster_array[:, :, :9] = raster_array[:, :, :9] / 10000.0

        blue_band = raster_array[:, :, 0]  # Blue
        red_band = raster_array[:, :, 2]   # Red
        nir_band = raster_array[:, :, 6]   # NIR

        # Compute vegetation indices
        ndvi = np.where(np.isnan(nir_band) | np.isnan(red_band), np.nan, 
                        (nir_band - red_band) / np.maximum(nir_band + red_band, 1e-10))
        evi = np.where(np.isnan(nir_band) | np.isnan(red_band) | np.isnan(blue_band), np.nan, 
                    2.5 * (nir_band - red_band) / np.maximum(nir_band + 6 * red_band - 7.5 * blue_band + 1, 1e-10))
        savi = np.where(np.isnan(nir_band) | np.isnan(red_band), np.nan, 
                        (nir_band - red_band) / np.maximum(nir_band + red_band + 0.5, 1e-10) * 1.5)

        # Normalize each index to be between 0 and 1, handling NaNs
        ndvi_normalized = np.where(np.isnan(ndvi), np.nan, ndvi / np.nanmean(ndvi))
        evi_normalized = np.where(np.isnan(evi), np.nan, evi / np.nanmean(evi))
        savi_normalized = np.where(np.isnan(savi), np.nan, savi / np.nanmean(savi))

        # Combine them into a single normalization factor while maintaining spatial dimensions
        biomass_norm_factor = np.nanmean(np.stack([ndvi_normalized, evi_normalized, savi_normalized]), axis=0)

        print("Applying model...\n")

        output_raster_array = self.apply_model_to_raster(dl_model, raster_array)
        if output_raster_array is None:
            return

        # Apply the biomass normalization factor
        output_raster_array[:, :, 0] *= biomass_norm_factor  # Woody biomass
        output_raster_array[:, :, 1] *= biomass_norm_factor  # Herbaceous biomass

        output_raster_array = np.clip(output_raster_array, 0, 1)

        sum_bands = np.sum(output_raster_array, axis=2, keepdims=True)
        sum_bands[sum_bands == 0] = 1  # Prevent division by zero
        output_raster_array = output_raster_array / sum_bands
        output_raster_array = output_raster_array.astype('float32')
        output_raster_array = np.nan_to_num(output_raster_array, nan=0)

        print("Success. Saving output raster...\n")

        output_raster_path = os.path.join(output_base_path, "vegetation_categories_raster.tif")

        with rasterio.open(input_raster_path) as src:
            profile = src.profile

        profile.update(
            count=3,  # Update the number of bands to 3
            dtype=output_raster_array.dtype,
            crs=self.vegetation_raster.crs if hasattr(self, 'vegetation_raster') else profile['crs']
        )

        print(f"Profile before saving: {profile}")
        print(f"Output raster shape: {output_raster_array.shape}")

        try:
            with rasterio.open(output_raster_path, 'w', **profile) as dst:
                for band_index in range(output_raster_array.shape[2]):
                    dst.write(output_raster_array[:, :, band_index], band_index + 1)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save the output raster: {e}")
            return

        self.vegetation_raster = rasterio.open(output_raster_path)  # Initialize the vegetation_raster attribute
        self.vegetation_raster_array = self.vegetation_raster.read()
        print(f"Vegetation category raster saved at: {output_raster_path}\n")
        messagebox.showinfo("Success", "Vegetation category raster saved successfully.")
        self.update_data_table_with_vegetation()

    def raster_to_array(self, raster_path):
        try:
            with rasterio.open(raster_path) as src:
                raster_array = src.read()
                raster_array = np.moveaxis(raster_array, 0, -1)  # Move the band axis to the last dimension
            return raster_array
        except rasterio.errors.RasterioIOError:
            messagebox.showerror("Error", "Failed to read the raster file. Please check the file path and try again.")
            return None

    def apply_model_to_raster(self, model, raster_array):
        if raster_array is None:
            return None

        height, width, bands = raster_array.shape
        flattened_raster = raster_array.reshape(-1, bands)
        predictions = model.predict(flattened_raster)
        predictions_normalized = predictions / predictions.sum(axis=1, keepdims=True)
        output = predictions_normalized.reshape(height, width, 3)
        return output

    def update_data_table_with_vegetation(self):
        if not hasattr(self, 'vegetation_raster_array'):
            messagebox.showerror("Error", "Vegetation raster not loaded.")
            return

        veg_categories = ['Woody fraction', 'Herbaceous fraction', 'Bare fraction']
        veg_data = {category: [] for category in veg_categories}

        for idx, row in self.voronoi_polygons_gdf.iterrows():
            mask = geometry_mask([row.geometry], transform=self.vegetation_raster.transform, invert=True, out_shape=self.vegetation_raster_array.shape[1:])
            for i, category in enumerate(veg_categories):
                category_band = self.vegetation_raster_array[i, :, :]
                veg_fraction = category_band[mask].mean()
                veg_data[category].append(veg_fraction)

        for category in veg_categories:
            self.add_vegetation_row(category, veg_data[category])

        self.data_frame.update_idletasks()  # Ensure layout is recalculated


    def add_vegetation_row(self, category, data):
        row_data = [category] + [f"{value:.2f}" for value in data]
        row_index = len(self.data_frame.grid_slaves()) // (len(self.site_names) + 2)  # Adjust for the number of columns including the plot column
        self.add_data_row(row_data, row_index)

    def acquire_sentinel(self):
        try:
            # Authenticate Earth Engine
            if not self.authenticate_earthengine():
                return
            
            # Load boundaries (shapefile) if they are not already loaded
            if not hasattr(self, 'boundaries_gdf'):
                messagebox.showerror("Error", "Boundaries not loaded. Please import boundaries first.")
                return
            
            gdf = self.boundaries_gdf
            gdf = gdf.to_crs("EPSG:4326")  # Ensure it's in WGS84

            # Union all geometries into one and get the bounding box
            geometry = gdf.geometry.unary_union
            geo_json = mapping(geometry)
            aoi = ee.Geometry.Polygon(geo_json['coordinates'])

            # Get the user-defined parameters from the GUI
            start_date = self.year_entry.get() + '-' + self.month_entry.get() + '-01'  # Use the year from the input and set the start date to January 1st
            end_date = self.year_entry.get() + '-' + self.month_entry.get() + '-30'    # Set the end date to December 31st
            cloud_cover = int(self.cloud_entry.get())

            # Load and process the Sentinel-2 dataset for the entire AOI
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate(start_date, end_date) \
                .filterBounds(aoi) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover)) \
                .median()

            # Select the RGB and additional bands
            image = collection.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12'])

            # Get the bounding box of the entire AOI
            bbox = geometry.bounds

            # Split the bounding box into smaller tiles for downloading
            boxes = self.split_bbox(bbox, max_size=0.05)  # Adjust max_size for each tile in degrees

            out_images = []
            for i, bbox in enumerate(boxes):
                geo_json = mapping(bbox)
                tile_aoi = ee.Geometry.Polygon(geo_json['coordinates'])
                total_tiles = len(boxes)
                print(f"Downloading tile {i + 1} out of {total_tiles}...")

                # Export each tile image
                out_file = f'{output_base_path}/sentinel_2_image_tile{i}.tif'
                geemap.ee_export_image(image, filename=out_file, scale=10, region=tile_aoi, file_per_band=False)
                out_images.append(out_file)

            # Merge the downloaded tiles
            src_files_to_mosaic = []
            for out_file in out_images:
                src = rasterio.open(out_file)
                src_files_to_mosaic.append(src)

            mosaic, out_trans = merge(src_files_to_mosaic)

            # Write the mosaic to a new file
            mosaic_meta = src_files_to_mosaic[0].meta.copy()
            mosaic_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "count": mosaic.shape[0],
                "compress": "lzw"  # Adding compression to reduce file size
            })

            final_image_path = f'{output_base_path}/sentinel_2_image.tif'
            with rasterio.open(final_image_path, "w", **mosaic_meta) as dest:
                dest.write(mosaic)

            # Close and delete individual tiles
            for src in src_files_to_mosaic:
                src.close()

            for out_file in out_images:
                os.remove(out_file)

            messagebox.showinfo("Success", f"Download and stitching complete. Image saved as {final_image_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to download and stitch Sentinel-2 imagery: {e}")

    def authenticate_earthengine(self):
        try:
            # Attempt to initialize Earth Engine
            ee.Initialize()
            messagebox.showinfo("Success", "Already authenticated with Google Earth Engine!")
        except ee.EEException as e:
            # If initialization fails, prompt the user to authenticate
            messagebox.showinfo("Earth Engine Authentication", "Please authenticate with Google Earth Engine.")
            
            try:
                # Use ee.Authenticate() to handle the authentication
                ee.Authenticate()  # This opens a browser window for the user to authenticate
                ee.Initialize()  # Try initializing again after authentication
                messagebox.showinfo("Success", "Google Earth Engine authentication successful!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to authenticate Earth Engine: {e}")
                return False
        except Exception as e:
            # Handle any other unforeseen exceptions
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            return False

        return True

    def split_bbox(self, bbox, max_size):
        min_x, min_y, max_x, max_y = bbox
        x_splits = int((max_x - min_x) // max_size) + 1
        y_splits = int((max_y - min_y) // max_size) + 1

        boxes = []
        for i in range(x_splits):
            for j in range(y_splits):
                new_min_x = min_x + i * max_size
                new_max_x = min_x + (i + 1) * max_size
                new_min_y = min_y + j * max_size
                new_max_y = min_y + (j + 1) * max_size

                new_max_x = min(new_max_x, max_x)
                new_max_y = min(new_max_y, max_y)

                boxes.append(box(new_min_x, new_min_y, new_max_x, new_max_y))

        return boxes

    def download_user_guide(self): 
        content = """
        ########################################################################################################################################################################################
        ### Spatially and Temporally Explicit Herbivory Monitoring in Water-limited Savannas (STEHM)
        ### Desktop Application: Brief User Guide
        ### Manuel Weber    
        ########################################################################################################################################################################################

        STEHM is implemented using two desktop applications, built using tkinter (Lundh, 1999) and auto-py-to-exe (Vollebregt, 2024). During long computations, the graphical user interface freezes, and progress is displayed in the console that opens additionally to the interfaces. Before implementing STEHM, the models need to be assessed on their local performance and possibly fine-tuned accordingly. Additionally, land managers should carefully consider the assumptions outlined in Table 1. Ecological heterogeneity within the area of interest may lead to differences between waterpoints for parameters that are considered equivalent, which would lead to a violation of the assumptions. 
        The apps require the following input to run:
        •	A shapefile (.shp) of the study area in EPSG:4326
        •	A comma-separated values file (.csv) with all waterpoint names and coordinates in EPSG:4326 stored in columns named “Site”, “Longitude” and “Latitude”. This will serve as a baseline to delimit waterpoint “catchment areas”, and therefore should include waterpoints without camera coverage.
        •	Camera trap images organized in folders named after the waterpoints: The directory supplied to the script to run the YOLO detection model should contain a number of subfolders with images, each of which corresponds to one waterpoint. A single subfolder can contain images from several cameras placed at the same waterpoint since the metadata of the images is used to delete duplicates, provided the date and time settings are synchronized.
        The camera trap imagery classification is performed using STEHM_v1.0_classifier.exe. The user clicks on “Select image folder” to navigate to the camera trap image folder. After setting the confidence threshold for the classification (default value: 0.75), the user clicks on “Classify images” to launch the process. An identical interface may pop up, which needs to be closed for the process to go ahead. The processing of the camera trap images represents the main computational bottleneck. Once completed, the output is saved as a comma-separated values file (.csv) under the desired name and location. The file contains, for each image, the path, the timestamp, the detection probability, and the count for each target species. The user can copy images to species folders by checking the boxes next to the species names in the interface and clicking on “Copy images to folders for selected species”.
        The rest of the analysis is performed in STEHM_v1.0.exe. The interface of the app is partitioned as follows: At the top left are the commands, arranged as buttons in three frames. The user progresses through the desired operations from top to bottom and from left to right. In the process, a table is created below the command section. The table is progressively filled with the output from the computations. The rows in the table serve as layers in the plot placed in the right half of the interface. The different layers can be viewed dynamically in the plot by using the checkboxes next to the table.
        Using the buttons in the “Step 1: Initialization” frame, the user creates an output folder named “run_YYYY-MM-DD” at the desired location by clicking on “Select general output directory”. Using the buttons “Import boundaries” and “Import waterpoints”, the user loads the boundary shapefile and the waterpoint comma-separated values file as described above, based on which the monitoring units delimited and their area displayed as a first row in the data table.
        The user then moves to the “Step 2: Camera trap images” frame. The output from the detection analysis performed using STEHM_v1.0_classifier.exe is converted to drinking events and feeding units using the “Compute RAIs” button. For duplicates to be effectively filtered out, all cameras at the same waterpoint need to have their time and date synchronized. The relative abundances of all species and feeding units are added as new rows to the data table. These rows can be visualized in the plot by using the checkboxes next to the table.
        The “Step 3: Vegetation” frame is used to contextualize herbivore abundance to biomass availability. After filling in the desired month, year, and maximum cloud cover, the “Acquire image” button downloads the median of bands 2, 3, 4, 5, 6, 7, 8, 11, and 12 of a Sentinel-2 image using the Google Earth Engine API (Gorelick et al., 2017). Upon first use, a browser window opens for authentication with the user’s Google account. Once downloaded, the “Run model” button applies the vegetation category model to the raster to produce an output with three bands, containing the cover fractions of woody (band 1) and herbaceous vegetation (band 2), as well as bare ground (band 3). New rows are added to the data table containing the vegetation category cover fractions and the feeding units detected per camera trap day and hectare of woody or herbaceous vegetation. Alternatively, the “Run model” button can be used to directly predict the growth forms from a locally saved Sentinel-2 product.
        After successfully performing all operations, the output folder should contain:
        •	“full_image_annotations.csv”: A comma-separated values file with the paths to all images alongside the probability for each target species to be present in the image, produced by the first app.
        •	“RAI.csv”: A comma-separated values file containing relative abundance indices for all waterpoints.
        •	“voronoi_polygons.shp” (and associated files): A shapefile in EPSG:4326 containing the Voronoi polygons is obtained by assigning each point in the area of interest to its nearest waterpoint.
        •	“sentinel_2_image.tif”: The downloaded Sentinel-2 product.
        •	“vegetation_categories_raster.tif”: The raster as predicted by the vegetation category model.
        •	“results.csv”: A comma-separated values file containing the feeding unit estimates contextualized to vegetation availability as computed from the vegetation model.
        •	“absolute_feeding_units.png” and “relative_feeding_units_per_area_of_biomass.png”: Bar plots to visualize the output.
        Questions can be directed to mweber1120@gmail.com.
        """
        
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write(content)
            messagebox.showinfo("Success", f"File saved to: {file_path}")

    def download_annotation_template(self):
        # Define the content of the CSV file
        data = {
            "Name": [
                "path\\to\\image.JPG", "path\\to\\image.JPG", "path\\to\\image.JPG",
                "path\\to\\image.JPG", "path\\to\\image.JPG", "path\\to\\image.JPG",
                "path\\to\\image.JPG", "path\\to\\image.JPG"
            ],
            "Date_Time": [
                "2024:07:20 10:32:34", "2024:07:20 10:32:50", "2024:07:20 10:33:16",
                "2024:07:20 10:33:52", "2024:07:20 10:34:17", "2024:07:20 10:34:30",
                "2024:07:20 21:35:53", "2024:07:20 21:36:45"
            ],
            "eland_confidence": [0, 0, 0, 0, 0, 0, 0, 0],
            "elephant_confidence": [0, 0, 0, 0, 0, 0, 0, 0],
            "giraffe_confidence": [
                0.979528904, 0.981678545, 0.980355859, 0.974310756,
                0.982983708, 0.972526371, 0, 0
            ],
            "impala_confidence": [0, 0, 0, 0, 0, 0, 0, 0],
            "kudu_confidence": [0, 0, 0, 0, 0, 0, 0, 0],
            "oryx_confidence": [0, 0, 0, 0, 0, 0, 0, 0],
            "rhino_confidence": [0, 0, 0, 0, 0, 0, 0.983135223, 0.982473671],
            "wildebeest_confidence": [0, 0, 0, 0, 0, 0, 0, 0],
            "zebra_confidence": [0, 0, 0, 0, 0, 0, 0, 0],
            "springbok_confidence": [0, 0, 0, 0, 0, 0, 0, 0],
            "eland_count": [0, 0, 0, 0, 0, 0, 0, 0],
            "elephant_count": [0, 0, 0, 0, 0, 0, 0, 0],
            "giraffe_count": [1, 1, 1, 1, 1, 1, 0, 0],
            "impala_count": [0, 0, 0, 0, 0, 0, 0, 0],
            "kudu_count": [0, 0, 0, 0, 0, 0, 0, 0],
            "oryx_count": [0, 0, 0, 0, 0, 0, 0, 0],
            "rhino_count": [0, 0, 0, 0, 0, 0, 1, 1],
            "wildebeest_count": [0, 0, 0, 0, 0, 0, 0, 0],
            "zebra_count": [0, 0, 0, 0, 0, 0, 0, 0],
            "springbok_count": [0, 0, 0, 0, 0, 0, 0, 0]
        }
        
        # Create a DataFrame from the data
        df = pd.DataFrame(data)
        
        # Open file dialog for saving the CSV file
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        
        if file_path:
            # Save the DataFrame as a CSV file
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"File saved to: {file_path}")
        else:
            messagebox.showerror("Error", "No file selected or invalid path.")

    def download_waterpoint_template(self):
        # Define the content of the CSV file
        data = {
            "Site": ["Site 1", "Site 2", "Site 3", "Site 4", "Site 5", "Site 6"],
            "Longitude": [17.132, 17.055, 17.059, 17.072, 17.121, 17.132],
            "Latitude": [-18.728, -18.764, -18.717, -18.742, -18.78, -18.736]
        }
        
        # Create a DataFrame from the data
        df = pd.DataFrame(data)
        
        # Open file dialog for saving the CSV file
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        
        if file_path:
            # Save the DataFrame as a CSV file
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"File saved to: {file_path}")
        else:
            messagebox.showerror("Error", "No file selected or invalid path.")

    def export_plot(self): 
        # Collect metadata from the user
        metadata = self.get_plot_metadata()

        # Open a file dialog to select where to save the file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save plot as"
        )
        
        # Check if a file path was provided (i.e., the user didn't cancel the dialog)
        if file_path:
            # Create a copy of the figure to add metadata without altering the GUI plot
            fig_copy = self.fig  # Duplicate the figure object
            ax_copy = fig_copy.gca()  # Get the axis of the duplicated figure

            # Add metadata to the copied plot
            ax_copy.set_title(metadata.get('title', ''), fontsize=15, pad=20)
            ax_copy.annotate(f"Author: {metadata.get('author', '')}", xy=(0.99, 0.01), xycoords='axes fraction',
                            fontsize=10, ha='right', va='bottom', color='gray')
            ax_copy.annotate(f"Date: {metadata.get('date', '')}", xy=(0.99, 0.05), xycoords='axes fraction',
                            fontsize=10, ha='right', va='bottom', color='gray')
            ax_copy.annotate(metadata.get('description', ''), xy=(0.5, -0.1), xycoords='axes fraction',
                            fontsize=10, ha='center', va='top', color='gray', wrap=True)

            # Save the copied figure to the specified file path
            fig_copy.savefig(file_path, format='png', bbox_inches='tight')

            # Optionally, close the copied figure to free up memory
            plt.close(fig_copy)

    def get_plot_metadata(self):
        def on_submit():
            metadata['title'] = title_var.get()
            metadata['author'] = author_var.get()
            metadata['date'] = date_var.get()
            metadata['description'] = description_var.get()
            metadata_dialog.destroy()

        metadata = {}
        metadata_dialog = tk.Toplevel(self.root)
        metadata_dialog.title("Plot Metadata")

        tk.Label(metadata_dialog, text="Title:").grid(row=0, column=0, padx=10, pady=5)
        title_var = tk.StringVar()
        tk.Entry(metadata_dialog, textvariable=title_var).grid(row=0, column=1, padx=10, pady=5)

        tk.Label(metadata_dialog, text="Author:").grid(row=1, column=0, padx=10, pady=5)
        author_var = tk.StringVar()
        tk.Entry(metadata_dialog, textvariable=author_var).grid(row=1, column=1, padx=10, pady=5)

        tk.Label(metadata_dialog, text="Date:").grid(row=2, column=0, padx=10, pady=5)
        date_var = tk.StringVar()
        tk.Entry(metadata_dialog, textvariable=date_var).grid(row=2, column=1, padx=10, pady=5)

        tk.Label(metadata_dialog, text="Description:").grid(row=3, column=0, padx=10, pady=5)
        description_var = tk.StringVar()
        tk.Entry(metadata_dialog, textvariable=description_var).grid(row=3, column=1, padx=10, pady=5)

        tk.Button(metadata_dialog, text="Submit", command=on_submit).grid(row=4, column=0, columnspan=2, pady=10)

        metadata_dialog.transient(self.root)
        metadata_dialog.grab_set()
        self.root.wait_window(metadata_dialog)

        return metadata

    '''
    def classify_images(self):
        def browse_folder():
            messagebox.showinfo("Information", "Please provide the path to a directory with folders that correspond to waterpoints. Only images in waterpoint folders will be classified.")
            folder_path = filedialog.askdirectory()
            return folder_path
        
        image_path = browse_folder()
        if not image_path:
            return

        # Define paths and threshold
        output_csv = os.path.join(output_base_path, 'full_image_annotations.csv')
        threshold = float(self.prob_thresh_entry.get()) if self.prob_thresh_entry.get() else 0.75

        # Run the yolo_inference.py script as a subprocess
        try:
            subprocess.run([sys.executable, "Scripts/STEHM_v2.0_yolo_inference.py", image_path, output_base_path, script_dir, str(threshold)], check=True)
            messagebox.showinfo("Success", f"YOLO classification completed. Results saved to {output_csv}.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"YOLO classification failed: {e}")
    '''

    def compute_rais(self):
        # Call the RAI calculation method
        self.RAI()

        # Read the RAI CSV file
        rai_file = f"{output_base_path}/RAI.csv"
        if not os.path.exists(rai_file):
            messagebox.showerror("Error", "RAI file not found.")
            return

        self.rai_df = pd.read_csv(rai_file)

        # Update the data display with RAIs
        self.update_data_with_rais()

        print("RAIs computation and table update completed.\n")

    def add_class_checkbox(self, class_name, row_index):
        if class_name not in self.plot_vars:
            self.plot_vars[class_name] = tk.BooleanVar()
        
        # Place the checkbox in the first column (column 0)
        plot_checkbox = ttk.Checkbutton(self.data_frame, variable=self.plot_vars[class_name], command=self.update_voronoi_plot_based_on_checkboxes)
        plot_checkbox.grid(row=row_index, column=0, sticky='nsew')  # Place in column 0

    def update_data_with_rais(self):
        headers = ["Plot"] + ["Sites"] + self.site_names  # Include 'Plot' as the first header

        # Clear previous data
        for widget in self.data_frame.winfo_children():
            widget.grid_forget()  # Clears the grid without destroying the widget
            widget.destroy()       # Destroys the widget

        # Create headers
        for col_index, col in enumerate(headers):
            width = 25 if col_index == 1 else 7
            label = ttk.Label(self.data_frame, text=col, borderwidth=1, relief="solid", width=width)
            label.grid(row=0, column=col_index, sticky='nsew')

        row_index = 1
        # Add area row
        if hasattr(self, 'voronoi_polygons_gdf'):
            areas = self.voronoi_polygons_gdf['Area (ha)'].tolist()
            self.add_data_row(['Area [ha]'] + [f"{area:.2f}" for area in areas], row_index)
            
            # Ensure only one checkbox for Area [ha]
            if 'Area [ha]' not in self.plot_vars:
                plot_var = tk.BooleanVar(value=True)
                self.plot_vars['Area [ha]'] = plot_var
            plot_checkbox = ttk.Checkbutton(self.data_frame, variable=self.plot_vars['Area [ha]'], command=self.update_voronoi_plot)
            plot_checkbox.grid(row=row_index, column=0, sticky='nsew')  # Place checkbox in the first column
            row_index += 1

        # Add RAI rows for relevant columns only (from time_range_days onwards)
        relevant_columns = [
            ('time_range_days', 'Coverage [days]'), 
            ('Grazing_units_absolute_selective', 'Abs. selective grazing'), 
            ('Browsing_units_absolute_selective', 'Abs. selective browsing'), 
            ('Grazing_units_absolute_bulk', 'Abs. bulk grazing'), 
            ('Browsing_units_absolute_bulk', 'Abs. bulk browsing')
        ] + [(col, f"{col.split('_', 1)[-1].capitalize()} [number/day]") for col in self.rai_df.columns if col.startswith('RAI_')]

        name_mapping = {
            'RAI_eland': 'Eland [number/day]',
            'RAI_elephant': 'Elephant [number/day]',
            'RAI_giraffe': 'Giraffe [number/day]',
            'RAI_impala': 'Impala [number/day]',
            'RAI_kudu': 'Kudu [number/day]',
            'RAI_oryx': 'Oryx [number/day]',
            'RAI_rhino': 'Rhino [number/day]',
            'RAI_wildebeest': 'Wildebeest [number/day]',
            'RAI_zebra': 'Zebra [number/day]',
            'RAI_springbok': 'Springbok [number/day]'
        }

        for class_name, display_name in relevant_columns:
            if class_name in name_mapping:
                display_name = name_mapping[class_name]
            rai_row = [display_name]  # Start with an empty string for the checkbox column
            for site in self.site_names:
                if site in self.rai_df['Waterpoint'].values:
                    rai_value = self.rai_df.loc[self.rai_df['Waterpoint'] == site, class_name].values[0]
                    rai_row.append(f"{rai_value:.2f}")
                else:
                    rai_row.append("")
            self.add_data_row(rai_row, row_index)
            self.add_class_checkbox(class_name, row_index)
            row_index += 1

    def update_voronoi_plot(self):
        self.update_voronoi_plot_based_on_checkboxes()

    def get_polygon_color(self, site, class_name):
        df = None
        if hasattr(self, 'rai_df') and class_name in self.rai_df.columns:
            df = self.rai_df
        elif hasattr(self, 'relative_feeding_units_df') and class_name in self.relative_feeding_units_df.columns:
            df = self.relative_feeding_units_df

        if df is not None:
            site_data = df[df['Waterpoint'] == site]
            if not site_data.empty:
                max_value = df[class_name].max()
                value = site_data[class_name].values[0]
                if not pd.isna(value) and value != 0:
                    normalized_value = value / max_value
                    rgba_color = plt.cm.Reds(normalized_value) if normalized_value > 0 else (0.5, 0.5, 0.5, 0.7)
                    return rgba_color[:3] + (0.7,)
        return None  # No color if no values

    def update_voronoi_plot_based_on_checkboxes(self):
        # Clear existing plot
        self.ax.clear()

        # Always plot boundaries first
        if hasattr(self, 'boundaries_gdf'):
            self.boundaries_gdf.boundary.plot(ax=self.ax, edgecolor='black')

        # Always plot waterpoints
        if hasattr(self, 'waterpoints_gdf'):
            self.waterpoints_gdf.plot(ax=self.ax, color='black', markersize=5)
            for x, y, label in zip(self.waterpoints_gdf.geometry.x, self.waterpoints_gdf.geometry.y, self.site_names):
                self.ax.text(x, y, label, fontsize=8, color='black')

        # Plot vegetation layers first as background
        vegetation_layers = {'Woody fraction': 0, 'Herbaceous fraction': 1, 'Bare fraction': 2}
        colormaps = {'Woody fraction': 'Greens', 'Herbaceous fraction': 'Greens', 'Bare fraction': 'YlOrBr'}

        for class_name, band_index in vegetation_layers.items():
            if self.plot_vars.get(class_name, tk.BooleanVar()).get() and hasattr(self, 'vegetation_raster_array'):
                band_data = self.vegetation_raster_array[band_index]
                extent = self.get_raster_extent()

                # Mask NaN values explicitly
                masked_data = np.ma.masked_invalid(band_data)

                # Normalize the data based on quantiles, excluding NaN values
                quantiles = np.nanquantile(band_data, [0.02, 0.98])
                norm = mcolors.Normalize(vmin=quantiles[0], vmax=quantiles[1])

                # Set a specific color for NaN values if desired
                cmap = plt.get_cmap(colormaps[class_name])
                cmap.set_bad(color='white')  # NaNs will appear as white

                # Plot the masked data
                self.ax.imshow(masked_data, extent=extent, cmap=cmap, norm=norm, interpolation='nearest', alpha=1.0)

        # Plot Voronoi polygons or other layers based on the checkbox values
        if hasattr(self, 'voronoi_polygons_gdf'):
            for idx, row in self.voronoi_polygons_gdf.iterrows():
                site = row['Site']
                outline = self.plot_vars.get('Area [ha]', tk.BooleanVar()).get()
                filled = False
                color = None

                for class_name, var in self.plot_vars.items():
                    if var.get() and class_name not in vegetation_layers:
                        color = self.get_polygon_color(site, class_name)
                        if color:
                            filled = True
                            break

                if outline and not filled:
                    self.voronoi_polygons_gdf.loc[[idx]].plot(ax=self.ax, edgecolor='black', facecolor='none', alpha=0.5)
                elif filled and color:
                    self.voronoi_polygons_gdf.loc[[idx]].plot(ax=self.ax, color=color, edgecolor='black', alpha=0.5)

        self.ax.set_aspect('auto')
        self.fig_canvas.draw()

    def get_image_metadata(self, image_path):
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == "DateTimeOriginal":
                    return value
        except Exception as e:
            print(f"Error extracting metadata for {image_path}: {e}\n")
        return None

    def organize_images(self):
        global results_df

        if results_df.empty:
            csv_file_path = os.path.join(output_base_path, 'full_image_annotations.csv')
            if os.path.exists(csv_file_path):
                results_df = pd.read_csv(csv_file_path)
                print(f"Loaded classification results from {csv_file_path}\n")
            else:
                messagebox.showerror("Error", "No classification results found. Perform classification first.")
                return

        threshold = float(self.prob_thresh_entry.get()) if self.prob_thresh_entry.get() else 0.75
        selected_species = [species for species, var in self.plot_vars.items() if var.get() and species.startswith('RAI_')]

        if not selected_species:
            messagebox.showerror("Error", "No species selected.")
            return

        for species in selected_species:
            species_name = species.split('_', 1)[-1]
            species_confidence = f"{species_name}_confidence"
            
            filtered_df = results_df[results_df[species_confidence] >= threshold]

            for image_path in filtered_df['Name']:
                # Extract the site name as the set of characters between the last two "/" or "\"
                site_name = os.path.normpath(image_path).split(os.sep)[-2]
                
                # Create species folder, then site folder within it
                species_folder = os.path.join(output_base_path, species_name)
                site_folder = os.path.join(species_folder, site_name)

                if not os.path.exists(site_folder):
                    os.makedirs(site_folder)

                try:
                    shutil.copy(image_path, site_folder)
                except Exception as e:
                    print(f"Error moving image {image_path}: {str(e)}\n")

        messagebox.showinfo("Success", "Images have been organized into species and site folders.")

    def RAI(self):
        def parse_datetime(date_str):
            formats = [
                "%Y:%m:%d %H:%M:%S",  # Original format
                "%Y/%m/%d %H:%M:%S",  # Common format
                "%Y-%m-%d %H:%M:%S",  # ISO format
                "%d/%m/%Y %H:%M",     # European format with no seconds
                "%Y:%m:%d %H:%M",     # Similar to original format but without seconds
                "%d-%m-%Y %H:%M:%S",  # European format with dashes
                "%m/%d/%Y %H:%M:%S",  # US format with slashes
                "%m-%d-%Y %H:%M:%S",  # US format with dashes
                "%d %b %Y %H:%M:%S",  # European format with abbreviated month
                "%d %B %Y %H:%M:%S",  # European format with full month
                "%d-%b-%Y %H:%M:%S",  # European format with abbreviated month and dashes
                "%d-%B-%Y %H:%M:%S",  # European format with full month and dashes
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None

        messagebox.showinfo("Warning", "All cameras at the same waterpoint need to have their time synchronized for duplicates to be effectively excluded.")

        global output_base_path
        if not output_base_path:
            messagebox.showerror("Error", "Output directory not set.")
            return

        
        # Check if full_image_annotations.csv already exists in the output directory
        target_csv_path = f"{output_base_path}/full_image_annotations.csv"
        if not os.path.exists(target_csv_path):
            try:
                # Open file dialog for user to select the CSV file
                csv_file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
                if not csv_file_path:
                    messagebox.showerror("Error", "No file selected.")
                    return
                # Copy the entire CSV to the target location
                shutil.copy(csv_file_path, target_csv_path)
                print(f"CSV file copied to {target_csv_path}\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy CSV file: {e}")
                return

        # Load herbivory data
        herbivory_data = {
            "Species": ["zebra", "wildebeest", "oryx", "eland", "elephant", "impala", "rhino", "giraffe", "kudu", "springbok"],
            "Grazing unit": [1.32, 1, 1.1, 2, 9.8, 0.3, 3.1, 3.2, 0.8, 0.3],
            "Browsing unit": [1.59, 1.21, 1.36, 2.4, 11.78, 0.4, 3.76, 3.8, 1, 0.37],
            "Selectivity": [0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
        }
        herbivory_df = pd.DataFrame(herbivory_data)
        species_names = herbivory_df["Species"].tolist()

        df = pd.read_csv(f"{output_base_path}/full_image_annotations.csv")
        df = df[df['Date_Time'] != "NA"]
        df['Date_Time'] = df['Date_Time'].astype(str).apply(parse_datetime)
        df = df.dropna(subset=['Date_Time'])
        df['Folder'] = df['Name'].apply(lambda x: os.path.basename(os.path.dirname(x)))

        threshold = float(self.prob_thresh_entry.get()) if self.prob_thresh_entry.get() else 0.75

        # Initial Event Delimitation (site- and species-specific)
        df = df.sort_values(by=['Folder', 'Date_Time'])
        df['Event'] = None

        for folder_name, group in df.groupby('Folder'):
            for species in species_names:
                species_conf_col = f"{species}_confidence"
                species_count_col = f"{species}_count"
                species_group = group[group[species_conf_col] > threshold]
                event_id = 0
                last_time = None
                for idx, row in species_group.iterrows():
                    if last_time is None or (row['Date_Time'] - last_time).total_seconds() > 1500:
                        event_id += 1
                    df.at[idx, 'Event'] = event_id
                    last_time = row['Date_Time']

        # Additional Condition for "elephant" and "rhino"
        for species in ["elephant", "rhino"]:
            for folder_name, group in df.groupby('Folder'):
                species_events = group[group[f"{species}_confidence"] > threshold]
                all_events = species_events['Event'].unique()
                new_event_id = 0
                merged_events = set()
                for event in all_events:
                    if event in merged_events:
                        continue
                    event_rows = species_events[species_events['Event'] == event]
                    if len(event_rows) == 0:
                        continue
                    next_event = event + 1
                    while next_event in all_events:
                        next_event_rows = species_events[species_events['Event'] == next_event]
                        if len(next_event_rows) == 0:
                            next_event += 1
                            continue
                        if len(group[(group['Date_Time'] > event_rows['Date_Time'].max()) & 
                                    (group['Date_Time'] < next_event_rows['Date_Time'].min()) & 
                                    (group[f"{species}_confidence"] <= threshold)]) >= 4:
                            break
                        merged_events.add(next_event)
                        event_rows = pd.concat([event_rows, next_event_rows])
                        next_event += 1
                    new_event_id += 1
                    df.loc[(df['Folder'] == folder_name) & (df['Event'].isin(event_rows['Event'].unique())), 'Event'] = new_event_id

        # Counting Individuals
        final_counts = {}
        for folder_name, group in df.groupby('Folder'):
            final_counts[folder_name] = {}
            for species in species_names:
                species_count_col = f"{species}_count"
                species_conf_col = f"{species}_confidence"
                species_group = group[group[species_conf_col] > threshold]
                event_max_counts = species_group.groupby('Event')[species_count_col].max()
                final_counts[folder_name][species] = event_max_counts.sum()

        # Calculating Time Range
        results = pd.DataFrame.from_dict(final_counts, orient='index').reset_index().rename(columns={'index': 'Waterpoint'})

        for folder_name, group in df.groupby('Folder'):
            min_time = group['Date_Time'].min()
            max_time = group['Date_Time'].max()
            time_range_days = (max_time - min_time).total_seconds() / (60 * 60 * 24)
            results.loc[results['Waterpoint'] == folder_name, 'time_range_days'] = time_range_days
    
        # Calculating RAIs
        for species in species_names:
            results[f'RAI_{species}'] = results.apply(lambda row: row.get(species, 0) / row['time_range_days'] if row['time_range_days'] > 0 else 0, axis=1)

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
        print(f"Relative abundance and feeding indices saved to: {csv_file}\n")
        self.save_absolute_feeding_units_plot(results, output_base_path)

    def save_absolute_feeding_units_plot(self, results, output_base_path):
        sorted_results = results.sort_values(by='Waterpoint')

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
        ax.set_yticklabels(sorted_results['Waterpoint'])
        ax.set_xlabel('Feeding units detected per camera trap day')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{abs(int(x))}"))  # Remove "-" and round the labels
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
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize='small', handlelength=1, handleheight=1, markerscale=0.8)

        plot_path = os.path.join(output_base_path, 'absolute_feeding_units.png')
        plt.savefig(plot_path)
        plt.close(fig)

    def relative_feeding_units_plot(self):
        results = pd.read_csv(f"{output_base_path}/results.csv")

        results = results.dropna(subset=[
            'Grazing_units/ha_herbaceous_vegetation_selective', 
            'Browsing_units/ha_woody_vegetation_selective', 
            'Grazing_units/ha_herbaceous_vegetation_bulk', 
            'Browsing_units/ha_woody_vegetation_bulk'
        ])

        # Sort results by waterpoint name alphabetically
        sorted_results = results.sort_values(by='Waterpoint')

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
        ax.set_yticklabels(sorted_results['Waterpoint'])
        ax.set_xlabel('Feeding units per camera trap day and ha of woody or herbaceous vegetation')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{abs(x):.2f}"))
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
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize='small', handlelength=1, handleheight=1, markerscale=0.8)

        plot_path = os.path.join(output_base_path, 'relative_feeding_units.png')
        plt.savefig(plot_path)
        
    def output(self):
        print("Computing final results and visualizations...\n")

        global output_base_path
        if not output_base_path:
            messagebox.showerror("Error", "Output directory not set.")
            return
        
        # File paths
        rai_file = f"{output_base_path}/RAI.csv"
        
        if not os.path.exists(rai_file):
            messagebox.showerror("Error", "RAI file not found.")
            return

        rai_df = pd.read_csv(rai_file)

        # Ensure Waterpoint column is treated as string
        rai_df['Waterpoint'] = rai_df['Waterpoint'].astype(str)

        vegetation_categories = ['Woody fraction', 'Herbaceous fraction', 'Bare fraction']
        veg_means = {category: [] for category in vegetation_categories}

        for idx, row in self.voronoi_polygons_gdf.iterrows():
            mask = geometry_mask([row.geometry], transform=self.vegetation_raster.transform, invert=True, out_shape=self.vegetation_raster_array.shape[1:])
            for i, category in enumerate(vegetation_categories):
                category_band = self.vegetation_raster_array[i, :, :]
                veg_fraction = category_band[mask].mean()
                veg_means[category].append(veg_fraction)

        veg_means_df = pd.DataFrame(veg_means)
        veg_means_df.index = self.voronoi_polygons_gdf['Site'].astype(str)

        # Merge RAI and vegetation data
        relative_feeding_units_df = pd.merge(rai_df, veg_means_df, left_on='Waterpoint', right_index=True)

        # Add area information to the DataFrame
        relative_feeding_units_df = pd.merge(relative_feeding_units_df, self.voronoi_polygons_gdf[['Site', 'Area (ha)']], left_on='Waterpoint', right_on='Site')

        # Calculate relative feeding units
        relative_feeding_units_df['Grazing_units/ha_herbaceous_vegetation_selective'] = (
            relative_feeding_units_df['Grazing_units_absolute_selective'] / 
            (relative_feeding_units_df['Herbaceous fraction'] * 
            relative_feeding_units_df['Area (ha)'])
        )
        relative_feeding_units_df['Browsing_units/ha_woody_vegetation_selective'] = (
            relative_feeding_units_df['Browsing_units_absolute_selective'] / 
            (relative_feeding_units_df['Woody fraction'] *
            relative_feeding_units_df['Area (ha)'])
        )
        relative_feeding_units_df['Grazing_units/ha_herbaceous_vegetation_bulk'] = (
            relative_feeding_units_df['Grazing_units_absolute_bulk'] / 
            (relative_feeding_units_df['Herbaceous fraction'] * 
            relative_feeding_units_df['Area (ha)'])
        )
        relative_feeding_units_df['Browsing_units/ha_woody_vegetation_bulk'] = (
            relative_feeding_units_df['Browsing_units_absolute_bulk'] / 
            (relative_feeding_units_df['Woody fraction'] *
            relative_feeding_units_df['Area (ha)'])
        )

        output_file = f"{output_base_path}/results.csv"
        relative_feeding_units_df.to_csv(output_file, index=False)
        print(f"Relative feeding units and vegetation categories results table saved to: {output_file}\n")

        self.relative_feeding_units_df = relative_feeding_units_df

        self.relative_feeding_units_plot()

    def add_relative_feeding_units_to_table(self):
        metrics = [
            ('Grazing_units/ha_herbaceous_vegetation_selective', 'Rel. selective grazing'), 
            ('Browsing_units/ha_woody_vegetation_selective', 'Rel. selective browsing'), 
            ('Grazing_units/ha_herbaceous_vegetation_bulk', 'Rel. bulk grazing'), 
            ('Browsing_units/ha_woody_vegetation_bulk', 'Rel. bulk browsing')
        ]

        for metric, display_name in metrics:
            row_data = [display_name]
            for site in self.site_names:
                if site in self.relative_feeding_units_df['Waterpoint'].values:
                    value = self.relative_feeding_units_df.loc[self.relative_feeding_units_df['Waterpoint'] == site, metric].values[0]
                    row_data.append(f"{value:.2f}")
                else:
                    row_data.append("")
            row_index = len(self.data_frame.grid_slaves()) // (len(self.site_names) + 2)
            self.add_data_row(row_data, row_index)
            self.add_class_checkbox(metric, row_index)

if __name__ == "__main__":
    root = tk.Tk()
    root.iconbitmap(os.path.join(script_dir, "Giraffe.ico"))
    app = HerbivoryApp(root)
    root.mainloop()