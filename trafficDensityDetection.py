import cv2 as cv 
import cv2.bgsegm
import numpy as np

# The threshold width and height of the cars
ww = 50
hh = 50


offset = 6

# The line y position on the window
line_y = 300


# All the detected objects in the video
detect = []

# No. of cars detected
cars = 0

# A function to find the center of a bounding box
def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Define the video cap
cap = cv.VideoCapture('street1.mp4')

# Create a background ssubtraction object
bgs = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()

    # Convert to grayscale
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Blurr to remove noise
    blur = cv.GaussianBlur(grey, (3, 3), 5)

    # Create the image subtraction
    img_sub = bgs.apply(grey)

    # Dilate the result for better detection
    dilate = cv.dilate(img_sub, np.ones((5, 5)))

    # Apply contours on the dilated results
    contour, h = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Create a line to count the crossing cars
    cv.line(frame, (25, line_y), (1200, line_y), (176, 130, 39), 2)

    # for each contoured object 
    for (i, c) in enumerate(contour):
        
        # Get the corners of the bounding rect around the contoured object
        (x, y, w, h) = cv.boundingRect(c)

        # Check if the object crosses the threshold to be detected
        valid_contour = (w >= ww) and (h >= hh)
        if not valid_contour:
            continue

        # Draw a rect around the valid contour
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find the center of the object from the given corners
        center = find_center(x, y, w, h)

        # Add the center of the object to the detected objects
        detect.append(center)

        # Mark the center with a red circle
        cv.circle(frame, center, 4, (0, 0, 255), -1)

        # for each detected center in detect
        for (x, y) in detect:

            # If the center crossed the line
            if (y < (line_y + offset)) and (y > (line_y - offset)):

                # Add 1 to the number of cars
                cars += 1

                # Change the color of the line to orange
                cv.line(frame, (25, line_y), (1200, line_y), (0, 127, 255), 3)

                # Remove the object from the detect list and print the number of cars
                detect.remove((x, y))

                print('no. cars detected: '+ str(cars))


    # Show the number of vehicles on the window
    cv.putText(frame, "VEHICLE COUNT : " + str(cars), (70, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 4)

    # 'Original video with the detections' window
    cv.imshow("Original Video", frame)

    # Detector's result
    cv.imshow(" Detector ", dilate)



    if cv.waitKey(18) == ord('q'):

        break        

cap.release()
cv.destroyAllWindows()
    
