import os
import cv2
import numpy as np

imageDir="D:\FrangiNet\FrangiNet\data\CHASEDB1\images"
imageDir="D:\\FrangiNet\\FrangiNet\\data\\Stare\\images"
Destination = "D:\\FrangiNet\\FrangiNet\\data\\Stare\\jpg"

AllImages = os.listdir(imageDir)
for i in AllImages:
    NowPath = os.path.join(imageDir, i)
    DestinationFileName = i[:-4] + ".jpg"
    NowImage = cv2.imread(NowPath)
    FullDestinationPath = os.path.join(Destination, DestinationFileName)
    #Mask = np.ones(NowImage.shape)
    cv2.imwrite(FullDestinationPath, NowImage)
    # NowImageTmp = np.sum(NowImage, axis=2)
    # Mask[NowImageTmp < 2] = 0
    # cv2.imshow('img',Mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #a = 1
    #b = 1
    # src = NowImage
    #
    # src = cv2.GaussianBlur(src, (3, 3), 0)
    # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("binary", binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # #image, contours, hierachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # image, contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for i in range(len(contours)):
    #     rrt = cv2.fitEllipse(contours[i])
    #     cv2.ellipse(src, rrt, (0, 0, 255), 2, cv2.LINE_AA)
    #     x, y = rrt[0]
    #     cv2.circle(src, (np.int(x), np.int(y)), 4, (255, 0, 0), -1, 8, 0)
    # cv2.imshow("fit circle", src)