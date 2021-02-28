import cv2 as cv
import numpy as np

mov = cv.VideoCapture('./resources/lynv1.mp4')
sharpenKernel = np.full((3,3), -1)
sharpenKernel[2][2] = 8
gblurKernel = np.full((3,3), 1/9)
print(gblurKernel)

while (mov.isOpened()):
    ret,frame = mov.read()
    if ret == True:
        sharp = cv.filter2D(frame,-1,sharpenKernel)
        blur = cv.GaussianBlur(frame, (21,21), cv.BORDER_DEFAULT)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cv.imshow('Frame', gray)      
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
mov.release()
cv.destroyAllWindows()


