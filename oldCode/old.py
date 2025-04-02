# this tracking app gets the pixel values of certain colors and averganges the centers to get the centers,
#this spesific one gets the point nearest to the last one os within a "neighborhood" of it so if there is background noise in the picture it wont affect it
#made by Abel message gonzale4@ualberta.ca for questions

#notice: this script is not very user freindly be warned

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.cluster import DBSCAN #pip3 install scikit-learn
import cv2
#gets array of all frames
def openFiles():
    return glob(r'frames\*.png')

VIDEO = r'Test1.mp4'
#type eatch color your looking for: typlicaly get the darkest and the lighets pixels on the points for the values
# color1 = [[35,48],[60,90],[130,200],"blue"] #RedRange,GreenRange,BlueRange,Name
# color2 = [[15,80],[121,200],[80,120],"green"]
# color3 = [[180,255],[20,60],[53,90],"pink"]
# color4 = [[240,255],[210,255],[132,160],"yellow"]
# colors = [color1,color2,color3,color4]

# #make a parralell array for the inital point values
# PointCordanitesArrColor1 = [[0,167,1014]]#type inital vales of eatch point: frame = 0,xstart,ystart
# PointCordanitesArrColor2 = [[0,82,1483]]
# PointCordanitesArrColor3 = [[0,63,1483]]
# PointCordanitesArrColor4 = [[0,151,1008]]
# PointCordanitesArr = [PointCordanitesArrColor1,PointCordanitesArrColor2,PointCordanitesArrColor3,PointCordanitesArrColor4]

color1 = [[20,48],[26,60],[47,130],"blue"] #RedRange,GreenRange,BlueRange,Name
color2 = [[15,60],[121,140],[50,100],"green"]
color3 = [[180,255],[20,80],[53,101],"pink"]
color4 = [[210,255],[210,255],[60,80],"yellow"]
colors = [color1,color2,color3,color4]

#make a parralell array for the inital point values
PointCordanitesArrColor1 = [[0,1473,473]]#type inital vales of eatch point: frame = 0,xstart,ystart
PointCordanitesArrColor2 = [[0,1464,510]]
PointCordanitesArrColor3 = [[0,2294,595]]
PointCordanitesArrColor4 = [[0,2291,566]]
PointCordanitesArr = [PointCordanitesArrColor1,PointCordanitesArrColor2,PointCordanitesArrColor3,PointCordanitesArrColor4]



badFrames = ['bad frames:']

gridResolution = []
for i in range(1,len(openFiles())):
    #loops for eatch color in frame
    for c in range(len(colors)):
        
        image1 = plt.imread(openFiles()[i])
        
        #<--- Getting colors ---->
        lenancy = 5 # ± 5 RGB values that it conisders that color
        
        pointColor = [i]
        RedRange = colors[c][0]
        GreenRange = colors[c][1]
        BlueRange = colors[c][2]
        #making mask
        mask = ((RedRange[0]-lenancy < 255*image1[:, :, 0]) & (255*image1[:, :, 0] < RedRange[1]+lenancy) &  # Red Channel
                     (GreenRange[0]-lenancy < 255*image1[:, :, 1]) & (255*image1[:, :, 1] < GreenRange[1]+lenancy) &  # Green Channel
                     (BlueRange[0]-lenancy < 255*image1[:, :, 2]) & (255*image1[:, :, 2] < BlueRange[1]+lenancy))  # Blue Channel
        PixelPoints = np.argwhere(mask)

        # <--- Image data managing ---->
        pointsLoc = PixelPoints.tolist()
        mainGrid = []
        for y in range(image1.shape[0]):
            row = []
            for x in range(image1.shape[1]):
                row.append(0)
            mainGrid.append(row)
        xPixels = len(mainGrid[0])
        yPixels = len(mainGrid)
        gridResolution = [xPixels,yPixels]
        # Example data: array of x, y coordinates
        coordinates = np.array(pointsLoc)

        try:
            # <--- pixel averaging  ---->
            # Perform DBSCAN clustering
            # If clusters are too fragmented, increase eps.
            # If clusters merge too much, decrease eps.
            # If many points are labeled as noise, reduce min_samples
            dbscan = DBSCAN(eps=20, min_samples=1) #these settings worked well for me
            labels = dbscan.fit_predict(coordinates)
            
            # Calculate cluster centers
            unique_labels = set(labels)
            cluster_centers = []
            unique_labels = set(labels) - {-1}  # Exclude noise points (-1)
            cluster_centers = {}
            for label in unique_labels:
                cluster_points = coordinates[labels == label]  # Select points in the cluster
                center = cluster_points.mean(axis=0)  # Compute mean
                cluster_centers[label] = center
            dfCenters = []
            for label, center in cluster_centers.items():
                dfCenters.append(list(center))
            for j in range(len(dfCenters)):
                temp = dfCenters[j][0]
                dfCenters[j][0] = (dfCenters[j][1])
                dfCenters[j][1] = (temp)
        
            # <-------------------------->

            theresPoint = 0
            
            
            tolorance = 50 # ± 25 x,y values that it conisders enar the pervios pixel
            for k in range(len(dfCenters)):
                if ((dfCenters[k][0] >= PointCordanitesArr[c][i-1][1] - tolorance) and (dfCenters[k][0] <= PointCordanitesArr[c][i-1][1] + tolorance)) and ((dfCenters[k][1] >= PointCordanitesArr[c][i-1][2] - tolorance) and (dfCenters[k][1] <= PointCordanitesArr[c][i-1][2] + tolorance)):
                    # print(f'appended {colors[c][3]} in frame {i}')
                    pointColor.append(dfCenters[k][0])
                    pointColor.append(dfCenters[k][1])
                    saftey=[]
                    saftey.append(dfCenters[k][0])
                    saftey.append(dfCenters[k][1])
                    theresPoint += 1
            #gets the closet points   
            
            if theresPoint ==0: #error giving
                print(f'no {colors[c][3]} points', openFiles()[i])
                print(f'previous: {PointCordanitesArr[c][i-1][1:]}')
                print('Current: ',dfCenters)
                # ### image Visualization:
                # plt.imshow(image1)
                # # print(dfCenters)
                # for center in dfCenters:
                #     plt.scatter(center[0], center[1], c='purple', marker='x')  # Cluster center 
                # plt.show() 
                badFrames.append(i)
                pointColor.append(PointCordanitesArr[c][i-1][1])
                pointColor.append(PointCordanitesArr[c][i-1][2])
                   
            if theresPoint > 1:# for the progam not to break it considers 1 points within the tolorance
                print(f'more than 1 {colors[c][3]} point', openFiles()[i])
                ### image Visualization:
                # plt.imshow(image1)
                # # print(dfCenters)
                # for center in dfCenters:
                #     plt.scatter(center[0], center[1], c='purple', marker='x')  # Cluster center 
                # plt.show() 
                pointColor.append(saftey[0])
                pointColor.append(saftey[1])
                badFrames.append(i)

        except ValueError: # more errors
            print(f'no {colors[c][3]} Pixels found', openFiles()[i])
            ### image Visualization:
            plt.imshow(image1)
            plt.show()
            print('BAD frames Updated be sure to fix')
            pointColor.append(PointCordanitesArr[c][i-1][1])
            pointColor.append(PointCordanitesArr[c][i-1][2])
            badFrames.append(i)
        if pointColor == []:
            print('ERROR')
        PointCordanitesArr[c].append(pointColor) 
    print(f'\rFrame {i} done ', end='', flush=True)
print(PointCordanitesArr)

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
print(PointCordanites)
title = ['frames']
for color in colors:
    title.append(f'{color[3]} x')
    title.append(f'{color[3]} y')
PointCordanites.insert(0,title)
print(PointCordanites)
PointCordanites = pd.DataFrame(PointCordanites)
PointCordanites.to_csv('csvs/dataPixels.csv',header=False,index=False)

# gets csvs in this format:
#frames,blue x,blue y,green x,green y,pink x,pink y,yellow x,yellow y,,,,
#1,168.45679012345678,1015.3086419753087,83.7196261682243,1482.9672897196263,59.848214285714285,1480.0714285714287,149.98360655737704,1010.4262295081967,,,,