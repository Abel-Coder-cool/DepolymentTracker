import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.cluster import DBSCAN
import cv2

VIDEO = r'Test1.mp4'
badFrames = []
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
image = cv2.imread(r"frames/frame_0000.png")
# height, width = image.shape[:2]
# print(f"Width: {width}, Height: {height}")

resolution = [1080, 1980]
fps = 60
print(f"Video FPS: {fps}")

# Load tracking data
csv_file_path = r'csvs/dataPixels.csv'
df = pd.read_csv(csv_file_path)

# Extract points and convert to Cartesian coordinates
point1 = df.iloc[:, 1:3].to_numpy(dtype=np.float64) # blue (outside elbow)
point2 = df.iloc[:, 3:5].to_numpy(dtype=np.float64) # green (sensor head)
point3 = df.iloc[:, 5:7].to_numpy(dtype=np.float64) # red (rotary hinge)
point4 = df.iloc[:, 7:9].to_numpy(dtype=np.float64) # yellow (inside elbow)

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

vector1 = point3 - point1  
vector2 = point2 - point3  

for frame in range(len(vector1)):
    # Normalize the vectors
    v1 = vector1[frame] / np.linalg.norm(vector1[frame])
    v2 = vector2[frame] / np.linalg.norm(vector2[frame])

    # Compute the angle in degrees
    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    angles.append(180 - angle)

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