import cv2
import numpy as np
#from matplotlib import pyplot as plt

# https://forum.opencv.org/t/detecting-the-center-of-a-curved-thick-line-in-python-using-opencv/1909

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x), int(y)

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

    #index = [-1]
    #print(thin)
    #thin = np.delete(thin, index)
    #print(thin)
    cv2.imshow("thin", thin)

    # dind contours
    cnts = cv2.findContours(thin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    # obtain outer coordinates
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])

    mask = np.zeros_like(image)

    #roi_vertices = np.array([left, right, top, bottom], np.int32)
    #cv2.fillPoly(mask, [roi_vertices], (255, 255, 255))

    #cv2.imshow("mask", mask)

    # draw connecting line
    cv2.line(image, left, right, (0,0,255), 2)

    # find the point of intersection
    # b/w connecting lines and curve points
    l1 = line(left, right)
    l2 = line((bottom[0], top[1]), bottom)
    inter = intersection(l1, l2)

    #draw center curve intersection line
    cv2.line(image, inter, bottom, (255, 255, 0), 2)

    # draw line contours
    cv2.drawContours(image, [c], -1, (36, 255, 12), 2)

    return image

image = cv2.imread("curves.png")
output = detect(image)

cv2.imshow("final",output)
cv2.waitKey(0)
cv2.destroyAllWindows()
