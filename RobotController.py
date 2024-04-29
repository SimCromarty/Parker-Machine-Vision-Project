# This file determines the functionality for controlling a robot arm
# Simeon Cromarty Master's Project
  
def pick_and_place(pick_x, pick_y, pick_h, place_x, place_y, place_h):
    
    # Convert image coordinates to robot workspace coordinates
    robot_x, robot_y = image_to_robot_coordinates(pick_x, pick_y)
    robot_place_x, robot_place_y = image_to_robot_coordinates(place_x, place_y)
    
    # Send pick and place to robot
    print(f'Robot moving to pick coordinates: (X: {robot_x}, Y: {robot_y}, H: {pick_h})')
    print('Robot picking item...')
    print('---------------------------')
    print(f'Robot moving to place coordinates: (X: {robot_place_x}, Y: {robot_place_y}, H: {place_h})')
    print('Robot placing...')
    print('---------------------------')
    pass

def image_to_robot_coordinates(image_x, image_y):
    # Scaling transformation for x and y (robot to real world coordinates)
    transform_x = 1.5
    transform_y = 1.5
    
    # Sets robot coordinates from image coordinates
    robot_x = round((image_x * transform_x), 1)
    robot_y = round((image_y * transform_y), 1)
    
    return robot_x, robot_y
    