# Simeon Cromarty Master's Project - Filter Locator
# Ultralytics YOLO library documentation used to guide implementation (https://docs.ultralytics.com/)
from ultralytics import YOLO
import cv2
import time
import os

# Import image processing functions from ImageProcessing
import ImageProcessing as imgpro

# Import robot controller functions from RobotController 
import RobotController as robot

# Sets model to trained Filter Locator weights
model = YOLO('VisionProject/Trained_Weights/FilterLocatorWeights_New_S50.pt')
print('Model Loaded')

# Call image processing function
processed_image = imgpro.process_image()
if processed_image is not None:
    cv2.imshow("Processed Image for Model", processed_image) # Display processed image
else:
    print("No image was processed.")
  
# Undistort image to remove lens distortion  
undistorted_image = imgpro.undistort_image(processed_image)
          
# Analyses processed image
results = model(undistorted_image, conf=0.8)

# Lists to store results
cls = []
conf = []
xyxy = []
centres = []

# Extracts results from image analysis
if len(results) > 0:
    print('Successful detection')
    for result in results:
        boxes = result.boxes.cpu().numpy() # Each instance of box detection
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0] # Unpack coordinates of each box 
            centre_x = (x1 + x2) / 2     # Calculate centre point of x and y's
            centre_y = (y1 + y2) / 2
            centres.append((centre_x, centre_y)) # Adds centres to list for robot use
else:
    print('No detection')

# Variables for robot operation
pick_height = 10  # Set value to robot pick height
place_height = 10  # Set value to robot place height
place_x, place_y = 100, 100  # Set place coordinates of robot

# Total number of detections
detections = len(boxes)
print('Number of detections = ', detections)
print('Initiating robot ')
print('___________________________')

# Actions for each detection
for i in range(detections):
    
    centre_x, centre_y = centres[i] # Unpack centres list into x and y
    centre_x_rounded = round(centre_x, 1)   # Round the centre coordinates to 1dp
    centre_y_rounded = round(centre_y, 1)
    
    print('Filter ',i,' location')
    print(f'Centre coordinates: (X: {centre_x_rounded}, Y: {centre_y_rounded})') # Outputs x and y coordinates of each filter
    
    robot.pick_and_place(centre_x_rounded, centre_y_rounded, pick_height, place_x, place_y, place_height) # Initiates robot pick and place for centre point coordinates
    time.sleep(2)
    
print('Robot successfully completed ', detections, ' pick and place tasks')    

# Draw circles on the centre point of each detected filter
for centre in centres:
    centre_x, centre_y = int(centre[0]), int(centre[1])  # Convert to integer for pixel coordinates
    cv2.circle(undistorted_image, (centre_x, centre_y), radius=5, color=(0, 255, 0), thickness=-1) # Draw circle at each centre point

# Display the result
cv2.imshow('Detected Filters', undistorted_image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close all OpenCV windows

# Save images for accuracy analysis
saveFolder = 'VisionProject/FilterLocatorAnalysis/'
imgName = 'FilterLocatorResult.jpg'
cv2.imwrite(os.path.join(saveFolder, imgName), undistorted_image)