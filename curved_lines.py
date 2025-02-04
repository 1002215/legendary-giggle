import cv2
import numpy as np


# https://forum.opencv.org/t/detecting-the-center-of-a-curved-thick-line-in-python-using-opencv/1909


def detect(image):
    # some preprocessing
    thin = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imshow("gray", thin)
    thin = cv2.blur(thin, (10, 10))
    cv2.imshow("blur", thin)
    _, thin = cv2.threshold(thin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("threshold", thin)

    #vertices = np.array([[(100, 100), (400, 100), (400, 400), (100, 400)]], dtype=np.int32)

    #mask = np.zeros_like(thin)

    #ROI = cv2.fillPoly(mask, vertices, 255)

    #region_of_interest = cv2.bitwise_and(thin, ROI)


    # https://stackoverflow.com/questions/19222343/filling-contours-with-opencv-python
    contours, hierarchy = cv2.findContours(thin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    cv2.drawContours(image, contours, 0, (0,0,255), thickness= 1)
    cv2.drawContours(image, contours, 1, (0,0,255), thickness = 1)

    # thin image to find clear contours
    thin = cv2.ximgproc.thinning(thin, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    cv2.imshow("thin", thin)

    # dind contours
    cnts, _ = cv2.findContours(thin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # c = max(cnts, key=cv2.contourArea)

    cv2.drawContours(image, cnts, 1, (36, 255, 12), 2)
    #image = cv2.polylines(final, [vertices], True, (0,0,255), 1)
    return image

image = cv2.imread("Screenshot 2025-02-04 at 8.18.02 AM.png")
cv2.imshow("final", detect(image))
cv2.waitKey(0)
cv2.destroyAllWindows()
