# Simeon Cromarty Master's Project - Image Processing
# This file investigates image pre processing techniques available with OpenCV with the aim of increasing vision model reliability.
# All functions used are from OpenCV library documents (https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)

# Import libraries
import cv2
import numpy as np

def process_image():
    # Take image
    cap = cv2.VideoCapture(1)                                                                           
    ret, frame = cap.read()                                                                            
    if ret:
        print('Image Taken')
        cap.release()
    else:
        print("Failed to capture image")
        cap.release()
        return None

    # Load the original colour image
    original_image = frame
    image_for_drawing = original_image.copy()

    # Make a copy of the original image for grayscale processing
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Grayscale Image", gray_image)
    #cv2.waitKey(0)

    # Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    #cv2.imshow("Blurred Image", blurred_image)
    #cv2.waitKey(0)

    # Sobel edge detection
    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobely = cv2.convertScaleAbs(sobely)
    sobel_combined = cv2.bitwise_or(abs_sobelx, abs_sobely)
    _, thresholded = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)

    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=1)

    # Find contours from dilated edges
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image and show
    cv2.drawContours(image_for_drawing, contours, -1, (0, 255, 0), 2)
    #cv2.imshow('Contours from Sobel Edges', image_for_drawing)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Debug: Visualise contours and bounding boxes
    debug_image = original_image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #cv2.imshow("Contours and Bounding Boxes", debug_image)
    #cv2.waitKey(0)

    # Box dimensions to determine expected aspect ratio
    expected_aspect_ratio = 600 / 420.0  # Length of box divided by width of box
    aspect_ratio_tolerance = 0.8         # Tolerance to account for lens distortion

    # Image area for relative size calculation
    image_area = original_image.shape[0] * original_image.shape[1]
    min_area_ratio = 0.2  # Minimum expected area ratio of the box relative to the image
    max_area_ratio = 0.95   # Maximum expected area ratio of the box relative to the image

    # Filtering contours based on aspect ratio and relative size
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        contour_aspect_ratio = w / float(h)
        contour_area = cv2.contourArea(cnt)
        contour_area_ratio = contour_area / float(image_area)

        # Check both aspect ratio and relative size criteria
        if (abs(contour_aspect_ratio - expected_aspect_ratio) <= aspect_ratio_tolerance and
                min_area_ratio <= contour_area_ratio <= max_area_ratio):
            valid_contours.append(cnt) # If contour valid then add to valid contours list

    # If a valid contour detected, find largest of the valid contours to determine box
    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        largest_contour_image = original_image.copy()
        cv2.drawContours(largest_contour_image, [largest_contour], -1, (0, 0, 255), 3)
        #cv2.imshow("Largest Contour", largest_contour_image)
        #cv2.waitKey(0)

        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        polygon_image = original_image.copy()
        cv2.drawContours(polygon_image, [largest_contour], -1, (0, 0, 255), 2)
        cv2.drawContours(polygon_image, [approx_polygon], -1, (255, 0, 0), 3)
        #cv2.imshow("Contour vs Approximated Polygon", polygon_image)
        #cv2.waitKey(0)

        # Convert polygon to rectangle
        rectangle_image = original_image.copy()
        padding = 7 # Padding to make rectangle slightly larger than box
        x, y, w, h = cv2.boundingRect(approx_polygon)
        x, y, w, h = x - padding, y - padding, w + 2 * padding, h + 2 * padding
        x = max(x, 0)  # Ensure x is not negative
        y = max(y, 0)  # Ensure y is not negative
        w = min(w, original_image.shape[1] - x)  # Ensure w does not exceed image width
        h = min(h, original_image.shape[0] - y)  # Ensure h does not exceed image height
        cv2.rectangle(rectangle_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.imshow("Bounding Rectangle", rectangle_image)
        #cv2.waitKey(0)
        # Crop the original colour image using the bounding rectangle of the largest contour
        cropped_roi = original_image[y:y+h, x:x+w]
    else:                                       # Error to show no box found in image
        print("No valid box contour found.")

    # Polarising filter effect - increase contrast

    # Convert to YUV color space, Y channel represents luminance
    yuv_image = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2YUV)
    y_channel, u, v = cv2.split(yuv_image)

    # Equalise the histogram of the Y channel
    equalised_y = cv2.equalizeHist(y_channel)
    
    # Blend the equalised Y channel with the original Y channel
    Scale = 0.5  # Increases contrast by 50%
    blended_y = cv2.addWeighted(y_channel, 1 - Scale, equalised_y, Scale, 0)

    # Merge the equalised Y channel back and convert to BGR
    yuv_image = cv2.merge([blended_y, u, v])
    final_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    # Image resizing with zero padding for model input
    # Calculate the scaling factor
    height, width = final_image.shape[:2]
    scale = 640 / max(height, width)

    # Resize the image with the aspect ratio
    resized_image = cv2.resize(final_image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    # Calculate padding
    delta_w = 640 - resized_image.shape[1]
    delta_h = 640 - resized_image.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Add padding to the resized image
    resized_padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return resized_padded_image

def undistort_image(image):
    # Pre calculated camera matrix from camera calibration 
    camera_matrix = np.array([[532.47459279, 0, 317.71358595],
                              [0, 530.56380816, 235.16618199],
                              [0, 0, 1]], dtype=np.float32)

    # Pre calculated distortion coefficients from camera calibration
    dist_coeffs = np.array([0.06952565, -0.20359022, 0.00026753, -0.00114502, 0.10290753], dtype=np.float32)

    h, w = image.shape[:2]
    new_camera_matrix = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # Undistort the image
    undistorted_img = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    return undistorted_img
