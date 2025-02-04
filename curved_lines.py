import cv2
import numpy as np

# https://forum.opencv.org/t/detecting-the-center-of-a-curved-thick-line-in-python-using-opencv/1909



def detect(image):
    # some preprocessing
    thin = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thin = cv2.blur(thin, (10, 10))
    _, thin = cv2.threshold(thin, 220, 255, 0)
    cv2.imshow("before", thin)

    #contours = cv2.findContours(thin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.imshow("contours", contours)
    #cv2.drawContours(image, contours, -1, (255,0,0), 2)


    # thin image to find clear contours
    thin = cv2.ximgproc.thinning(thin, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    cv2.imshow("thin", thin)

    # dind contours
    cnts, _ = cv2.findContours(thin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(image, cnts[1], -1, (36, 255, 12), 2)

    return image

image = cv2.imread("curves.jpg")
output = detect(image)

cv2.imshow("final",output)
cv2.waitKey(0)
cv2.destroyAllWindows()
