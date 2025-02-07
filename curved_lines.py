import cv2
import numpy as np

# Emma Chetan Parallel Curved Lines and Centerline Project PWP

# Key features of the detect() function:
#   1. Takes in an image in format .jpg, .jpeg, .png
#   2. Image processing to reduce any noise using opencv functions
#      such as COLOR_RGB2GRAY (grayscale conversion), blur, adaptive
#      thresholding, findContours, drawContours, and .shape (returning
#      dimensions of an image).
#   3. The use of the Guo Hall thinning algorithm to detect the center line.
#   4. A raw video stream where for each frame, the parallel curved lines and
#      center line are drawn.

def detect(image):

    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply a blur to reduce noise on the image.
    blur = cv2.blur(gray, (5, 5), 0)

    #https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/
    # Use adaptive thresholding, where each pizel contributes to computing the
    # optimal T, or threshold, value.
    thresh = cv2.adaptiveThreshold(blur, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    # https://stackoverflow.com/questions/19222343/filling-contours-with-opencv-python
    # Find contours after applying thresholding. This should return the contours
    # of the parallel curved lines.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contour of one of the parallel curved lines in red.
    cv2.drawContours(image, contours, 2, (0,0,255), thickness= 10)

    # Draw the contours of the other parallel curved line in red.
    cv2.drawContours(image, contours , 1, (0,0,255), thickness = 10)

    # https://arxiv.org/pdf/1710.03025#:~:text=This%20paper%20proposes%20a%20sequential%20algorithm%20that%20is,and%203D%20patterns%20and%20showed%20very%20good%20results.
    # The thinning function uses the Guo Hall thinning algorithm, which returns a 1 pixel-wide,
    # connected, centered skeleton inside a given component, in this case the two curved
    # parallel lines.
    thin = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_GUOHALL)

    # Find the contours of the skeleton structure.
    cnts, _ = cv2.findContours(thin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours of the skeleton, or center line, structure.
    cv2.drawContours(image, cnts, 1, (36, 255, 12), 2)
    
    rows, columns = image.shape[0], image.shape[1]

    # Crop the image to get rid of reduntant detected contours.
    cropped = image[10:rows-10, 10:columns-10]

    return cropped

# Run a live video stream where the detect() function is applied to
# draw on the parallel curved lines and centerline.

# Opens the camera stream on the default camera
cap = cv2.VideoCapture(0)

if cap.isOpened():

    while True:

        try:

            # Read the frame from the video capture.
            success, frame = cap.read()

            # If the frame was not read correctly, break the loop to end the program.
            if success == True:

                # Display the frame.
                cv2.imshow("Emma Chetan Parallel Curved Lines and Centerline PWP", detect(frame))

                # Check if the user pressed 'q' to end the program.
                if cv2.waitKey(25) == ord('q'):

                    break

            else:

                break

        except Exception:

            continue

# Release the video capture and close all windows.
cap.release()

cv2.destroyAllWindows()
