# ## this tracking app gets the pixel values of certain colors and averganges the centers to get the centers,
# ##this spesific one gets the point nearest to the last one os within a "neighborhood" of it so if there is background noise in the picture it wont affect it
# ##made by Abel message gonzale4@ualberta.ca for questions
# ##notice: this script is not very user freindly be warned
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.cluster import DBSCAN
import cv2
import time
# ##gets array of all frames
def openFiles():
   return sorted(glob(r'frames/*.png'))

VIDEO = r'Small.mp4'
# ##type eatch color your looking for: typlicaly get the darkest and the lighets pixels on the points for the values
color1 = [[20,48],[26,60],[47,130],"blue"] #RedRange,GreenRange,BlueRange,Name
color2 = [[15,80],[121,190],[50,100],"green"]
color3 = [[180,255],[20,100],[53,110],"pink"]
color4 = [[200,255],[195,255],[60,100],"yellow"]
colors = [color1,color2,color3,color4]
# ##make a parralell array for the inital point values
PointCordanitesArrColor1 = [[0,699,236]]#type inital vales of eatch point: frame = 0,xstart,ystart
PointCordanitesArrColor2 = [[0,684,258]]
PointCordanitesArrColor3 = [[0,1111,272]]
PointCordanitesArrColor4 = [[0,1109,260]]
PointCordanitesArr = [PointCordanitesArrColor1,PointCordanitesArrColor2,PointCordanitesArrColor3,PointCordanitesArrColor4]
##type eatch color your looking for: typlicaly get the darkest and the lighets pixels on the points for the values
## color1 = [[35,48],[60,90],[130,200],"blue"] #RedRange,GreenRange,BlueRange,Name
## color2 = [[15,80],[121,200],[80,120],"green"]
## color3 = [[180,255],[20,60],[53,90],"pink"]
## color4 = [[240,255],[210,255],[132,160],"yellow"]
## colors = [color1,color2,color3,color4]
## #make a parralell array for the inital point values
## PointCordanitesArrColor1 = [[0,167,1014]]#type inital vales of eatch point: frame = 0,xstart,ystart
## PointCordanitesArrColor2 = [[0,82,1483]]
## PointCordanitesArrColor3 = [[0,63,1483]]
## PointCordanitesArrColor4 = [[0,151,1008]]
## PointCordanitesArr = [PointCordanitesArrColor1,PointCordanitesArrColor2,PointCordanitesArrColor3,PointCordanitesArrColor4]
badFrames = []
gridResolution = []
frame_files = openFiles()
for i in range(1,len(frame_files)):
   start_time_frame = time.time()
   #loops for eatch color in frame
   for c in range(len(colors)):
      
      
       image1 = cv2.imread(frame_files[i])  # Ensure correct scaling (0-255 uint8)
       image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert to RGB from OpenCV's default BGR
      
       #<--- Getting colors ---->
       lenancy = 5 # ± 5 RGB values that it conisders that color
      
       pointColor = [i]
       RedRange = colors[c][0]
       GreenRange = colors[c][1]
       BlueRange = colors[c][2]
       #making mask
       lower_bound = np.array([RedRange[0]-lenancy, GreenRange[0]-lenancy, BlueRange[0]-lenancy])
       upper_bound = np.array([RedRange[1]+lenancy, GreenRange[1]+lenancy, BlueRange[1]+lenancy])
       mask = cv2.inRange(image1, lower_bound, upper_bound)
       PixelPoints = np.column_stack(np.where(mask > 0))
       # <--- Image data managing ---->
       if PixelPoints.size == 0:
           print(f' no {colors[c][3]} points color range is not good enough', frame_files[i])
           print(f'previous: {PointCordanitesArr[c][i-1][1:]}')
           badFrames.append(i)
           pointColor.append(PointCordanitesArr[c][i-1][1])
           pointColor.append(PointCordanitesArr[c][i-1][2])
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
           tolorance = 15 #+-
           
           for center in dfCenters:
               if ((center[0] >= PointCordanitesArr[c][i-1][1] - tolorance) and (center[0] <= PointCordanitesArr[c][i-1][1] + tolorance)) and \
                   ((center[1] >= PointCordanitesArr[c][i-1][2] - tolorance) and (center[1] <= PointCordanitesArr[c][i-1][2] + tolorance)):
                    if theresPoint == 0:
                        pointColor.extend(center)
                    theresPoint += 1

           if theresPoint == 0: #error giving
               print(f' no {colors[c][3]} points in interval try chanceing tolorance or color range', frame_files[i])
               badFrames.append(i)
               pointColor.append(PointCordanitesArr[c][i-1][1])
               pointColor.append(PointCordanitesArr[c][i-1][2])
           elif theresPoint > 1:
               print(f'more than one {colors[c][3]} point in interval', frame_files[i])
               pointColor.extend(dfCenters[0])
               badFrames.append(i)
      
       PointCordanitesArr[c].append(pointColor) 
   end_time_frame = time.time()
   print(f'\rFrame {i} done estimated time: {((end_time_frame - start_time_frame) * (len(frame_files) - i)) / 60:.2f} minutes', end='', flush=True)

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

# gets csvs in this format:
#frames,blue x,blue y,green x,green y,pink x,pink y,yellow x,yellow y,,,,
#1,168.45679012345678,1015.3086419753087,83.7196261682243,1482.9672897196263,59.848214285714285,1480.0714285714287,149.98360655737704,1010.4262295081967,,,,

# Sam Leaders Part of the code modified by Abel
# --- Video Processing Functions ---
def get_fps(video_path):
    """Get FPS from video file."""
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

# --- Main Processing ---
# Get video properties
image = cv2.imread(r"frames/frame_0119.png")
height, width = image.shape[:2]
print(f"Width: {width}, Height: {height}")
resolution = [height, width]
fps = get_fps(VIDEO)
print(f"Video FPS: {fps}")

# Load tracking data
csv_file_path = r'csvs/dataPixels.csv'
df = pd.read_csv(csv_file_path)

# Extract points and convert to Cartesian coordinates
point1 = df.iloc[:, 1:3].to_numpy(dtype=np.float64)  # Blue
point2 = df.iloc[:, 3:5].to_numpy(dtype=np.float64)  # Green
point3 = df.iloc[:, 5:7].to_numpy(dtype=np.float64)  # Red
point4 = df.iloc[:, 7:9].to_numpy(dtype=np.float64)  # Yellow

ptarray = [point1, point2, point3, point4]
# convert from screen coordinates to cartesian coordinates
for point in ptarray:
    point[:,1] = resolution[1] - point[:,1]

# Handle bad frames
if len(badFrames) > 10:
    print(f'Warning: More than 10 bad frames detected\nBad Frames: {badFrames}')
else:
    print(f'Bad Frames: {badFrames}')
if len(badFrames) !=0:
    
    badFrames_zero_indexed = np.array(badFrames) - 1
    mask = np.ones(point1.shape[0], dtype=bool)
    mask[badFrames_zero_indexed] = False

    point1 = point1[mask]
    point2 = point2[mask]
    point3 = point3[mask]
    point4 = point4[mask]

angles = []
for frame in range(len(point1)):
    
    # ensure that the direction of the vector is consistent for both the outside and the inside
    outside_line = ptarray[0][frame] - ptarray[1][frame]
    outside_vector = (outside_line)/np.linalg.norm(outside_line)
    inside_line = ptarray[3][frame] - ptarray[2][frame]
    inside_vector = (inside_line)/np.linalg.norm(inside_line)
    angle = np.degrees(np.arccos(np.dot(outside_vector, inside_vector)))
    angles.append(angle)

times = np.arange(len(angles)) / fps
angular_velocity = np.gradient(angles, times)
angular_acceleration = np.gradient(angular_velocity, times)

# --- Save Results ---
results = pd.DataFrame({
    'Time (s)': times,
    'Angle (deg)': angles,
    'Angular Velocity (deg/s)': angular_velocity,
    'Angular Acceleration (deg/s²)': angular_acceleration
})

output_path = f"csvs/{VIDEO.replace('.mp4', '')}_Data.csv"
results.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
