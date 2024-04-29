# Import necessary libraries
from ultralytics import YOLO                                                                                
import cv2
import torch

class ObjectDetector:
    def __init__(self, model_path):
        # Initialise YOLO model
        self.model = YOLO(model_path)                                                                       # Sets model to specified weights path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'                                        # Checks if GPU available, if not CPU
        self.model.to(self.device)
        print('initialisation complete')

    def capture_image(self):
        # Initialise webcam
        cap = cv2.VideoCapture(1)                                                                           # Captures image of source 1 (webcam)
        ret, frame = cap.read()                                                                             # Capture a single frame 
        print('Image taken')
        cap.release()                                                                                       # Release webcam now frame is taken
        if not ret:                                                                                         # If frame not captured successfully, raise error
            raise IOError("Couldn't capture an image from the webcam")
        return frame                                                                                        # Return frame

    def analyse_image(self, img):
        # Run inference
        results = self.model(img)                                                                       # Run model on converted image
        print(type(results))
        return results                                                                                      # Return results

    def get_bounding_boxes(self, results):
        # Extract bounding box coordinates
        boxes = results.boxes.xyxy.cpu().numpy()                                                       # Assuming xyxy format, converts to CPU tensor and numpy array
        if len(boxes) > 0:
            return boxes                                                                                       # Return box coordinates
        else:
            print('no bounding boxes detected')
            return None

    def get_class_instances(self, results):
        # Count instances of each class
        class_counts = torch.bincount(results.boxes.cls.int())                                              # Counts each detection in results
        class_names = {i: self.model.names[i] for i in range(len(class_counts)) if class_counts[i] > 0}     # Creates dictionary for mapping class ID to names
        return class_names, class_counts                                                                    # Returns names and counts of each

    def plot_results(self, results, img):
        # Plot and show results on the original image
        annotated_image = results.plot(show=True, img=img)                                                  # Plots results onto original image
        cv2.imshow("Annotated Image", annotated_image)                                                      # Shows results plot in new window
        cv2.waitKey(0)                                                                                      # Wait for a key press to exit
        cv2.destroyAllWindows()                                                                             # Destorys all windows 

    def access_results_attributes(self, results):
        # Check if results is a list and iterate, else work with it directly
        if isinstance(results, list):
            for result in results:
                self.print_result_attributes(result)
        else:
            self.print_result_attributes(results)

    def print_result_attributes(self, result):
        # Access and print attributes of a single Results object
        print("Processing Results Object...")

        # Bounding Boxes
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.cpu().numpy()
            
            #xyxys = boxes.xyxy
            #for xyxy in xyxys:
                #cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), (0,0,255), 3))
            print("Bounding Boxes:\n", boxes)

        # Detected Class Names
        if hasattr(result, 'names'):
            detected_classes = [result.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()] if result.boxes is not None else []
            print("Detected Classes:", detected_classes)

        # Original Image
        if hasattr(result, 'orig_img'):
            print("Original Image Shape:", result.orig_img.shape)
            cv2.imshow('Original Image', result.orig_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print('No original image found')

        # Keypoints
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            print("Keypoints:", result.keypoints)

        # Masks
        if hasattr(result, 'masks') and result.masks is not None:
            print("Masks:", result.masks)

        # Class Probabilities
        if hasattr(result, 'probs') and result.probs is not None:
            print("Class Probabilities:", result.probs)

        # Image Path
        if hasattr(result, 'path'):
            print("Image Path:", result.path)

        print("\n")






def O_Ring():
    # O Ring Detection
    
    o_ring_detector = ObjectDetector(model_path='VisionProject/O_Ring_Train/weights/O_Ring_Weights.pt')     # Initialise
    img = o_ring_detector.capture_image()                                                                   # Capture image
    results = o_ring_detector.analyse_image(img)                                                            # Analyse image with model
    o_ring_detector.access_results_attributes(results)
    #boxes = o_ring_detector.get_bounding_boxes(results)                                                     # Extracts bounding boxes
    #class_names, class_counts = o_ring_detector.get_class_instances(results)                                # Gets name and count of each detection
    #o_ring_detector.plot_results(results, img)                                                              # Displays annotated image 
    #if len(boxes) > 0:                                                                                      # If there are bounding boxes, then an O-ring is detected
        #class_names, class_counts = o_ring_detector.get_class_instances(results)                            # Get class names and counts
        #o_ring_detector.plot_results(results, img)                                                          # Plot and display the detection results on the image
        #return boxes, class_names, class_counts                                                             # Return extracted results
    #else:
        #return None, None, None                                                                             # No O-ring detected, return None

def Filter():
    # Filter Detection
    filter_detector = ObjectDetector(model_path='VisionProject/Filter_Train/weights/Filter_Weights.pt')     # Initialise
    img = filter_detector.capture_image()                                                                   # Capture image
    results = filter_detector.analyse_image(img)                                                            # Analyse image with model
    boxes = filter_detector.get_bounding_boxes(results)                                                     # Extracts bounding boxes
    class_names, class_counts = filter_detector.get_class_instances(results)                                # Gets name and count of each detection
    filter_detector.plot_results(results, img)                                                              # Displays annotated image
    if len(boxes) > 0:                                                                                      # Check if any bounding boxes were detected (i.e., any filters were detected)
        class_names, class_counts = filter_detector.get_class_instances(results)                            # Get class names and counts
        filter_detector.plot_results(results, img)                                                          # Plot and display the detection results on the image
        return boxes, class_names, class_counts                                                             # Return extracted results
    else:
        return None, None, None                                                                             # No filters detected, return None for each detail


def main():
    #boxes, names, class_counts = O_Ring()    # Get results from O_Ring detection
    #print(boxes, names, class_counts)
    O_Ring()
    #if boxes:                                                    # If o_ring_boxes is not None and not empty (O-ring detected)
        #Filter()                                                        # Proceed to detect Filter
    #else:
        #print('No O-Ring detected')                                     # No O-ring detected, print message

if __name__ == "__main__":
    main()