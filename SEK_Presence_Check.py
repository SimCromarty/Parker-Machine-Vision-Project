# Simeon Cromarty Master's Project - SEK Presence Checker

from ultralytics import YOLO
import cv2

# This code should follow the presnece check logic for checking o-ring first, then for filters.

def detect_o_ring():
    # Sets model to O-Ring weights from training
    model = YOLO('VisionProject/Trained_Weights/O_Ring_Weights_New_S50.pt')
    print('O-Ring Model Loaded')
    
    # Take image
    cap = cv2.VideoCapture(1)                                                                           # Captures image from source 1 (webcam)
    ret, frame = cap.read()                                                                             # Capture a single frame 
    print('Image Taken')
    cap.release()    
    
    # Flags to track the presence of filters
    o_ring_detected = False
    
    # Predicts image
    results = model(frame, conf=0.85)
    
    # Lists to store results
    o_ring_cls = []
    o_ring_conf = []
    
    # Extract results
    if len(results) > 0:
        print('Successful detection')
        for result in results:
            # Each instance of detection
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                o_ring_cls.append(box.cls)
                o_ring_conf.append(box.conf)
            result.show()
    else:
        print('No detection')
    
    # Check each result in boxes
    for i in range(len(boxes)):
    
        if (o_ring_cls[i][0]) == 0:
            print('Detected O-Ring presence') # O-ring detected
            o_ring_detected = True            # Set presence flag to true
        print('Confidence for detection:',o_ring_conf[i][0])
    
    if (o_ring_detected):
        return True
    else:
        return False
        
        
def detect_filters():
    # Sets model to Filter weights from training
    model = YOLO('VisionProject/Trained_Weights/SEK_Filter_Weights_New_S50.pt')
    print('SEK Filter Model Loaded')
    
    # Take image
    cap = cv2.VideoCapture(1)                                                                           # Captures image from source 1 (webcam)
    ret, frame = cap.read()                                                                             # Capture a single frame 
    print('Image Taken')
    cap.release()   
    
    # Presence Flags
    main_filter_detected = False
    pre_filter_detected = False 
    
    # Predicts image
    results = model(frame, conf=0.85)
    
    # Lists to store results
    filter_cls = []
    filter_conf = []
    
    # Extract results
    if len(results) > 0:
        print('Successful detection')
        for result in results:
            #Each instance of detection
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                filter_cls.append(box.cls)
                filter_conf.append(box.conf)
            result.show()
        
    else:
        print('No detection')
    
    # Check each result in boxes
    for i in range(len(boxes)):
    
        if (filter_cls[i][0]) == 0:
            print('Detected Main Filter presence')
            main_filter_detected = True
        if (filter_cls[i][0]) == 1:
            print('Detected Pre Filter presence')
            pre_filter_detected = True
        print('Confidence for detection:',filter_conf[i][0])
    
    # Presence check logic
    if main_filter_detected and pre_filter_detected:
        return 'Both'
    elif main_filter_detected and not pre_filter_detected:
        return 'Main'
    elif not main_filter_detected and pre_filter_detected:
        return 'Pre'
    else:
        return 'None'
    
    
def PLC_confirmation(prompt):
    response = input(prompt)
    return response == '1'


while True:
    # O-Ring Detection Phase
    if PLC_confirmation("Enter 1 to signal o-ring has been placed, or any other key to exit: "):
        o_ring_detected = detect_o_ring()  # Capture the return value to check if the o-ring was detected
        if not o_ring_detected:  # Check if o-ring was not detected
            print("O-ring not detected. Please ensure it is correctly placed.")
            if not PLC_confirmation("Error detected. Enter 1 after correcting to retry, or any other key to exit: "):
                break  # Exit if the user does not confirm to retry
            continue  # Retry from the beginning if o-ring detection fails
        print("O-ring detected. Next detection: Filters")
    else:
        break  # Exit if user does not confirm o-ring placement

    # Filter Detection Phase
    filters_detected = False
    while not filters_detected:  # Loop until both filters are detected
        if PLC_confirmation("Enter 1 to signal filters have been placed, or any other key to exit: "):
            filter_status = detect_filters()
            if filter_status == "Both":
                print("Successful presence check. Both filters detected.")
                filters_detected = True  # Set a flag or use appropriate logic to break out of the filter loop
            else:
                error_message = "Missing filters: "
                if filter_status == "Main":
                    error_message += "Pre Filter"
                elif filter_status == "Pre":
                    error_message += "Main Filter"
                else:  # None detected
                    error_message += "Main and Pre Filters"
                print(error_message)
                print('To retry see next message')
        else:
            break  # Exit the filter loop if user does not confirm filter placement

    if filters_detected:  # Check if the entire process is completed successfully
        if not PLC_confirmation("Presence check complete. Enter 1 to continue next cycle, or any other key to exit: "):
            break  # Exit the main loop if the user does not want to repeat after successful detection
        else:
            o_ring_detected = False  # Reset detection flags for a new cycle
            filters_detected = False
    else:
        break  # Exit the main loop if filters are not detected and user chooses not to retry