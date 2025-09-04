# this tracking app gets the pixel values of certain colors and averganges the centers to get the centers,
#this spesific one gets the point nearest to the last one os within a "neighborhood" of it so if there is background noise in the picture it wont affect it
#made by Abel message gonzale4@ualberta.ca for questions
#notice: this script is not very user freindly be warned
#gets array of all frames
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import sys
from videotopngs import extract_frames
import os
from tkinter import Scrollbar, HORIZONTAL, VERTICAL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.cluster import DBSCAN  #pip3 install scikit-learn
import cv2
import time
import shutil
zoom_level = 1.0  # initial zoom
zoom_step = 0.1   # zoom increment
min_zoom = 0.1
max_zoom = 3.0

def display_image():
    """Display the current image at the current zoom level."""
    global tk_img
    if img:
        width = int(img.width * zoom_level)
        height = int(img.height * zoom_level)
        resized = img.resize((width, height), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)
        image_canvas.delete("all")  # Clear previous image
        image_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        image_canvas.config(scrollregion=(0, 0, width, height))

def zoom(event):
    """Zoom in or out with mouse wheel."""
    global zoom_level
    # Windows scroll: event.delta is multiple of 120
    if event.delta > 0:
        zoom_level = min(zoom_level + zoom_step, max_zoom)
    else:
        zoom_level = max(zoom_level - zoom_step, min_zoom)
    display_image()

def load_image_Inital():
    global img
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    if video_path:
        output_folder = "extracted_frames"
        extract_frames(video_path, output_folder)

        frame_path = os.path.join(output_folder, "frame_0000.png")
        if os.path.exists(frame_path):
            img = Image.open(frame_path).convert("RGB")
            display_image()
            image_canvas.bind("<Button-1>", get_pixel)
        else:
            printToTerminal("Error: frame_0000.png not found.")


def load_image(IMAGEfile):
    global img
    output_folder = "extracted_frames"
    frame_path = os.path.join(output_folder, IMAGEfile)
    if os.path.exists(frame_path):
            img = Image.open(frame_path).convert("RGB")
            display_image()
            image_canvas.bind("<Button-1>", get_pixel)

root = tk.Tk()
root.title("Point Data Getter")
root.geometry("1400x1000")
root.configure(bg="#d3d3d3")

# Global image objects
img = None
tk_img = None

# Global list to store points
point_list = []

def enter_point():
    data = entry.get()
    try:
        x, y, r, g, b = map(int, data.split(","))
        point_list.append((x, y, (r-20,r+20), (g-20,g+20), (b-20,b+20)))
        update_current_points()
    except ValueError:
        terminal_text.insert(tk.END, "Invalid input. Use: X, Y, R, G, B\n")

def update_current_points():
    # Clear the text box first
    current_points_text.delete("1.0", tk.END)

    display_items = []
    for idx, (x, y, r, g, b) in enumerate(point_list):
        # r, g, b are tuples of (low, high)
        current_points_text.insert(
            tk.END,
            f"{idx}: ({x}, {y}) - "
            f"R{r} G{g} B{b}\n"
        )
        display_items.append(f"Point {idx}: ({x}, {y})")

    # Update the combo box values
    point_selector["values"] = display_items
    if display_items:  # select the last added by default
        point_selector.current(len(display_items) - 1)

# Function to handle click and update entry
def get_pixel(event):
    if img:
            # Canvas scroll offsets
        canvas_x = image_canvas.canvasx(event.x)
        canvas_y = image_canvas.canvasy(event.y)
    
        # Convert from display coordinates to original image coordinates
        real_x = int(canvas_x / zoom_level)
        real_y = int(canvas_y / zoom_level)
        try:
            r, g, b = img.getpixel((real_x, real_y))
            coord_rgb = f"{real_x}, {real_y}, {r}, {g}, {b}"
            entry.delete(0, tk.END)
            entry.insert(0, coord_rgb)
            printToTerminal(f"Clicked at: ({real_x}, {real_y}) - RGB: ({r}, {g}, {b})")

        except IndexError:
            printToTerminal("Clicked outside of image bounds")

# --- UI Layout ---
#titel
entry_label = tk.Label(root, text=" Abel Point Tracker ", bg="#6AA7D9", fg="#FFFFFF",font=("Arial", 50))
entry_label.place(x=20, y= 20)

# Open the image
image = Image.open("AbelMoon - Copy.png")
image = image.resize((120, 150), Image.Resampling.LANCZOS)
photo = ImageTk.PhotoImage(image)
imagelabel = tk.Label(root, image=photo)
imagelabel.image = photo  # Keep a reference!
imagelabel.place(x=625, y=20, width=120, height=150)

# Top Button
top_button = tk.Button(root, text="pick video", bg="#ffd662", fg="#2b8bd0", font=("Arial", 10), command=load_image_Inital)
top_button.place(x=20, y= 100+40, width=100, height=30)

# Left Canvas for Image
# Image display frame
image_frame = tk.Frame(root, bg="#d3d3d3", bd=1, relief="solid")
image_frame.place(x=20, y= 100+80, width=720, height=660)

# Scrollbars
x_scroll = Scrollbar(image_frame, orient=HORIZONTAL)
y_scroll = Scrollbar(image_frame, orient=VERTICAL)
x_scroll.pack(side="bottom", fill="x")
y_scroll.pack(side="right", fill="y")

# Canvas with scrollbars
image_canvas = tk.Canvas(image_frame, bg="#d3d3d3", xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
image_canvas.pack(side="left", fill="both", expand=True)
# Bind zoom on scroll
image_canvas.bind("<MouseWheel>", zoom)  # Windows and Mac
image_canvas.bind("<Button-4>", zoom)    # Linux scroll up
image_canvas.bind("<Button-5>", zoom)    # Linux scroll down

x_scroll.config(command=image_canvas.xview)
y_scroll.config(command=image_canvas.yview)

def deletefiles():
     if os.path.exists("extracted_frames"):
        for filename in os.listdir("extracted_frames"):
            file_path = os.path.join("extracted_frames", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)   # remove file or shortcut
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # remove subfolder
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        printToTerminal(f"Cleared: {"extracted_frames"}",level="info")
    
# Right Panel
right_button = tk.Button(root, text="point Data", bg="#007bbf", fg="white", font=("Arial", 10))
right_button.place(x=880, y= 100+85, width=280, height=40)
right_button = tk.Button(root, text="Clean Up For New Video", bg="red", fg="black", font=("Arial", 10),command=deletefiles)
right_button.place(x=880, y= 100, width=280, height=40)


entry_label = tk.Label(root, text="X, Y, Red, Green, Blue", bg="#d3d3d3")
entry_label.place(x=800, y= 100+150)

entry = tk.Entry(root, font=("Arial", 12))
entry.place(x=800, y= 100+180, width=400, height=30)

enter_button = tk.Button(root, text="Enter Point", width=15, command=enter_point)
enter_button.place(x=950, y= 100+230, width=100, height=30)

current_points_label = tk.Label(root, text="CurrentPoints", bg="#d3d3d3")
current_points_label.place(x=800, y= 100+260)

current_points_text = tk.Text(root, height=5, width=30)
current_points_text.place(x=800, y= 100+290, width=500, height=50)

terminal_label = tk.Label(root, text="Terminal", bg="#d3d3d3")
terminal_label.place(x=800, y= 720)

terminal_text = tk.Text(root, height=10, width=50, font=("Comic Sans MS", 14))
terminal_text.place(x=800, y= 750, width=500, height=200)

# --- Right Panel continued ---
# Dropdown to pick point
pick_point_label = tk.Label(root, text="Pick Point:", bg="#d3d3d3", font=("Arial", 12))
pick_point_label.place(x=800, y=450)

point_selector = ttk.Combobox(root, state="readonly")
point_selector.place(x=900, y=450, width=200, height=30)

# Section Title
edit_label = tk.Label(root, text="Edit Color ranges", bg="#d3d3d3", font=("Arial", 14, "bold"))
edit_label.place(x=800, y=500)

# Column Labels
lower_label = tk.Label(root, text="Lower", bg="#d3d3d3", font=("Arial", 12))
upper_label = tk.Label(root, text="Upper", bg="#d3d3d3", font=("Arial", 12))
lower_label.place(x=950, y=530)
upper_label.place(x=1050, y=530)

# Row Labels + Entry fields
r_label = tk.Label(root, text="Red", bg="#d3d3d3", font=("Arial", 12))
g_label = tk.Label(root, text="Green", bg="#d3d3d3", font=("Arial", 12))
b_label = tk.Label(root, text="Blue", bg="#d3d3d3", font=("Arial", 12))
r_label.place(x=800, y=560)
g_label.place(x=800, y=600)
b_label.place(x=800, y=640)

# Entries for lower and upper ranges
r_lower = tk.Entry(root, width=5)
r_upper = tk.Entry(root, width=5)
g_lower = tk.Entry(root, width=5)
g_upper = tk.Entry(root, width=5)
b_lower = tk.Entry(root, width=5)
b_upper = tk.Entry(root, width=5)

r_lower.place(x=950, y=560, width=60)
r_upper.place(x=1050, y=560, width=60)
g_lower.place(x=950, y=600, width=60)
g_upper.place(x=1050, y=600, width=60)
b_lower.place(x=950, y=640, width=60)
b_upper.place(x=1050, y=640, width=60)

def on_point_selected(event):
    idx = point_selector.current()
    if idx >= 0:
        x, y, r_range, g_range, b_range = point_list[idx]
        # r_range, g_range, b_range are tuples like (low, high)
        r_lower.delete(0, tk.END); r_lower.insert(0, r_range[0])
        r_upper.delete(0, tk.END); r_upper.insert(0, r_range[1])
        g_lower.delete(0, tk.END); g_lower.insert(0, g_range[0])
        g_upper.delete(0, tk.END); g_upper.insert(0, g_range[1])
        b_lower.delete(0, tk.END); b_lower.insert(0, b_range[0])
        b_upper.delete(0, tk.END); b_upper.insert(0, b_range[1])

def apply_edits():
    idx = point_selector.current()
    if idx >= 0:
        try:
            r_range = (int(r_lower.get()), int(r_upper.get()))
            g_range = (int(g_lower.get()), int(g_upper.get()))
            b_range = (int(b_lower.get()), int(b_upper.get()))
            x, y, _, _, _ = point_list[idx]
            point_list[idx] = (x, y, r_range, g_range, b_range)
            update_current_points()
            printToTerminal(f"Updated point {idx}")
        except ValueError:
            printToTerminal("Invalid range values")

apply_button = tk.Button(root, text="Apply Changes", command=apply_edits)
apply_button.place(x=950, y=680, width=120, height=30)

point_selector.bind("<<ComboboxSelected>>", on_point_selected)


#print to terminal
terminal_text.tag_config("error", foreground="red")
terminal_text.tag_config("success", foreground="green")
terminal_text.tag_config("info", foreground="blue")
terminal_text.tag_config("normal", foreground="black")

def printToTerminal(msg, level="normal"):
    if isinstance(msg, list):
        for item in msg:
            terminal_text.insert(tk.END, str(item) + '\n', level)
            print(str(item))
    else:
        terminal_text.insert(tk.END, str(msg) + '\n', level)
        print(str(msg))
    terminal_text.see(tk.END)  

def clearTerminal():
    terminal_text.delete("1.0", tk.END)

def startProcces():
    def openFiles():
       return sorted(glob(r'extracted_frames/*.png'))
    ##type eatch color your looking for: typlicaly get the darkest and the lighets pixels on the points for the values
    colorName = ["point 0","point 1","point 2","point 3"]
    colors = []
    PointCordanitesArr=[]
    for i in range(len(point_list)):
        colors.append([[point_list[i][2][0],point_list[i][2][1]],[point_list[i][3][0],point_list[i][3][1]],[point_list[i][4][0],point_list[i][4][1]],colorName[i]])
        PointCordanitesArr.append([[0,point_list[i][0],point_list[i][1]]]) 

    printToTerminal(colors)
    printToTerminal(PointCordanitesArr)
    badFrames = []
    frame_files = openFiles()
    for i in range(1,len(frame_files)):
       start_time_frame = time.time()
       #loops for eatch color in frame
       for c in range(len(colors)):
           image1 = cv2.imread(frame_files[i])  # Ensure correct scaling (0-255 uint8)
           image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert to RGB from OpenCV's default BGR
          
           #<--- Getting colors ----> 
           pointColor = [i]
           RedRange = colors[c][0]
           GreenRange = colors[c][1]
           BlueRange = colors[c][2]
           #making mask
           lower_bound = np.array([RedRange[0], GreenRange[0], BlueRange[0]])
           upper_bound = np.array([RedRange[1], GreenRange[1], BlueRange[1]])
           
           mask = cv2.inRange(image1, lower_bound, upper_bound)
           PixelPoints = np.column_stack(np.where(mask > 0))
           # <--- Image data managing ---->
           if PixelPoints.size == 0:
               printToTerminal(f' no {colors[c][3]} points color range is not good enough\n'+ str(frame_files[i])+"change range and try again",level="error")
               printToTerminal(f'previous: {PointCordanitesArr[c][i-1][1:]}',level="error")
               load_image(str(frame_files[i]).split("\\")[-1])
               return
               
           else:
               # <--- pixel averaging  ---->
               dbscan = DBSCAN(eps=20, min_samples=1) #these settings worked well for me
               labels = dbscan.fit_predict(PixelPoints)
              
               # Calculate cluster centers
               unique_labels = set(labels)
               cluster_centers = {}
               for label in unique_labels:
                   cluster_points = PixelPoints[labels == label]  # Select points in the cluster
                   center = cluster_points.mean(axis=0)  # Compute mean
                   cluster_centers[label] = center
               dfCenters = [list(center) for center in cluster_centers.values()]
               for j in range(len(dfCenters)):
                   dfCenters[j][0], dfCenters[j][1] = dfCenters[j][1], dfCenters[j][0]
              
               # <-------------------------->
               theresPoint = 0
               tolorance = 25 #+-
               
               for center in dfCenters:
                   if ((center[0] >= PointCordanitesArr[c][i-1][1] - tolorance) and (center[0] <= PointCordanitesArr[c][i-1][1] + tolorance)) and \
                       ((center[1] >= PointCordanitesArr[c][i-1][2] - tolorance) and (center[1] <= PointCordanitesArr[c][i-1][2] + tolorance)):
                        if theresPoint == 0:
                            pointColor.extend(center)
                        theresPoint += 1
               if theresPoint == 0: #error giving
                   printToTerminal(f' no {colors[c][3]} points in interval try chanceing tolorance or color range\n'+ str(frame_files[i])+" change range and try again",level="error")
                   load_image(str(frame_files[i]).split("\\")[-1])
                   return
                   badFrames.append(i)
                   pointColor.append(PointCordanitesArr[c][i-1][1])
                   pointColor.append(PointCordanitesArr[c][i-1][2])
               elif theresPoint > 1:
                   printToTerminal(f'more than one {colors[c][3]} point in interval\n'+ str(frame_files[i])+"change range and try again",level="error")
                   load_image(str(frame_files[i]).split("\\")[-1])
                   return
                   pointColor.extend(dfCenters[0])
                   badFrames.append(i)
          
           PointCordanitesArr[c].append(pointColor)
       end_time_frame = time.time()
       printToTerminal(f'\rFrame {i} done estimated time: {((end_time_frame - start_time_frame) * (len(frame_files) - i)) / 60:.2f} minutes')
       clearTerminal()
    def transform_array(data):
       result = {}
       for sublist in data:
           for entry in sublist:
               idx = entry[0]
               if idx not in result:
                   result[idx] = [idx]
               result[idx].extend(entry[1:])
       return list(result.values())
    PointCordanites = transform_array(PointCordanitesArr)
    del PointCordanites[0]
    title = ['frames']
    for color in colors:
       title.append(f'{color[3]} x')
       title.append(f'{color[3]} y')
    PointCordanites.insert(0,title)
    PointCordanites = pd.DataFrame(PointCordanites)
    PointCordanites.to_csv('csvs/dataPixels.csv',header=False,index=False) #optimized by chat gpt 😎👍
    printToTerminal(f"Pixel Locations saved to csvs/dataPixels.csv",level="succes")

procces_button = tk.Button(root, text="startTracking", width=15, command=startProcces,bg="#0018cf",fg="white")
procces_button.place(x=950, y= 710, width=100, height=30)
root.mainloop()