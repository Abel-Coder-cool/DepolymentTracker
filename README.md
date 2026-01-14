--Note--
I made this for the AlbertaSat Ex-Alta3 project, but anyone is free to use the program. 
I created this program in school, so I might refine it in the future when I have more time.

--Requirements--
To run the program, you need Python 3 installed on your computer. You need to have pillow, pandas, numpy, matplotlib, scikit-learn, and opencv-python installed in addition to Python. I have made a run script called installRequirements_windows.bat for Windows and installRequirements_mac_linux.sh for Mac and Linux. To execute the file on Mac, you might need to make the .sh file an executable.

--Use--
You can start the program by the bat or .sh file labeled start program. After the program is open, use the pick video button and select the video you want to track the points of. If you want to choose a different video or, before closing the program, use the clean-up for a new video button. After the video is selected, you can use the mouse to select points of interest. This program runs by analyzing the colors in the picture, so make sure the points you pick stay that color for the entire video. This was made for tracking specific points in a deployment. When a point is selected, click the enter point button, then when all the points are picked, press the start tracking button. The status of the tracking will appear in the terminal that opened with the program. If the program gives an error, you might need to adjust the color of the point using the edit color change feature (these errors are covered a bit more in depth in the video). After the program runs, it will return the results of the tracking in the CSVs folder (the program will tell you where it saved the file and its name).

--Addidtonal comments--
-This program statistically means the color data's coordinates, so it works on non-circular colored points
-If the object traverses through the frame too quickly,> ~25 pixels/frame, the program will not be able to track it. For the specific use case I made this for, it made the program run better. In the future, I might improve this so it works with faster-moving points.

tutorial video: https://youtu.be/picQwi_75Aw (a bit out of date)
