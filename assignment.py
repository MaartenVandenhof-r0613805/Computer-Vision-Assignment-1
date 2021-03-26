import cv2 as cv
import numpy as np
import time

mov = cv.VideoCapture('../resources/60secondsT.mp4')
#mov = cv.VideoCapture('../resources/last20.mp4')
#mov = cv.VideoCapture('E:/Projects/resources/kotbal.mp4')
#mov = cv.VideoCapture(0)
width = int(mov.get(3))
height = int(mov.get(4))
out = cv.VideoWriter('../resources/finalv1.avi'),
cv.VideoWriter_fourcc('M','J','P','G'), 
30, (width, height))
template = cv.imread('../resources/template.jpg', 0)
tomato = cv.imread('../resources/tomato.png', cv.IMREAD_UNCHANGED)
tomato = cv.resize(tomato, (280,280))


#Initialize Kernels
sharpenKernel = np.full((3,3), -1)
sharpenKernel[2][2] = 8
normalKernel = np.full((3,3), 0)
normalKernel[2][2] = 1
sobelxKernel = np.matrix('-1,-2,-1; 0,0,0; 1,2,1')
sobelykernel = np.matrix('-1, 0, 1; -2, 0, 2; -1, 0, 1')

#Variables
changeFrame = False
startTime = time.time()
points = []

def inRange(x, y, width, height):
    if x < 1920 and width > 0 and y < 1080 and height > 0:
        return True
    else:
        return False

while (mov.isOpened()):
    ret,frame = mov.read()
    if ret == True:
        
        frames = mov.get(cv.CAP_PROP_FRAME_COUNT)
        fps = int(mov.get(cv.CAP_PROP_POS_MSEC))
        duration = (int(fps/(frames/59))/10)/196.6*60
        print(duration)
        
        #Frames
        showFrame = frame
        original = cv.filter2D(frame, -1, normalKernel)
        sharp = cv.filter2D(frame,-1,sharpenKernel)
        blur = cv.GaussianBlur(frame, (3,3), 0)
        blurBig = cv.GaussianBlur(frame, (9,9), 0)
        blurHuge = cv.GaussianBlur(frame, (15,15), 0)
        ##Frame to BGR Grayscale
        grayv1 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.cvtColor(grayv1, cv.COLOR_GRAY2BGR)
        ##Colorspaces
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        hsv = cv.cvtColor(gray, cv.COLOR_BGR2HSV)
        ##Threshold frame
        ###Ball threshold
        true_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_bound = np.array([30,55,0])
        upper_bound = np.array([70,255,255])
        lower_bound_rgb = np.array([50, 80, 0])
        upper_bound_rgb = np.array([255, 255, 255])
        ###Poster threshold
        lower_bound_poster = np.array([0,25,0])
        upper_bound_poster = np.array([90,255,255])
        ### Use blur for smoothing
        true_hsv_blur = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        true_hsv_hugeBlur = cv.cvtColor(blurHuge, cv.COLOR_BGR2HSV)
        ### Mask for range and Res for color
        ###Poster mask
        mask_poster = cv.inRange(true_hsv_hugeBlur, lower_bound_poster, upper_bound_poster)
        ###Ball mask
        mask_bgr = cv.inRange(frame, lower_bound_rgb, upper_bound_rgb)
        mask_bgr_inv = cv.bitwise_not(mask_bgr)
        mask_hsv = cv.inRange(true_hsv, lower_bound, upper_bound)
        mask_blur = cv.inRange(true_hsv_blur, lower_bound, upper_bound)
        res_bgr = cv.bitwise_and(frame, frame, mask=mask_bgr)
        res_hsv = cv.bitwise_and(frame, frame, mask=mask_hsv)
        res_blur = cv.bitwise_and(frame, frame, mask=mask_blur)
        ###Erode and dilate masks
        morf_kernel = np.ones((5,5), np.uint8)
        poster_diluted = cv.dilate(mask_poster, morf_kernel, iterations=20)
        hsv_eroded = cv.erode(mask_hsv, morf_kernel, iterations=1)
        hsv_dilute = cv.dilate(mask_hsv, morf_kernel, iterations=7)

        #Edge detection
        def sobelContour(frame, ksize, val):
            if val == "x":
                sframe = cv.Sobel(frame, cv.CV_64F, 1, 0, ksize=ksize)
            else:
                sframe = cv.Sobel(frame, cv.CV_64F, 0, 1, ksize=ksize)
            
            contours, hiearchy = cv.findContours(255-cv.convertScaleAbs(sframe), cv.RETR_TREE , cv.CHAIN_APPROX_SIMPLE)
            return contours
        gray_blur = cv.GaussianBlur(grayv1, (15,15), 0)
        ##Find contours poster
        contours_poster, h_poster = cv.findContours(255-poster_diluted, 1,2)
        canny_poster= cv.Canny(gray_blur, 0,80)

        #Mean Squared Error
        ##Flashy rectangle (now circle)
        if time.time() - startTime < 10:
            circleColor = (255,225,0)
        if time.time() - startTime >= 10 and time.time() - startTime < 30:
            circleColor = (0,225,255)
        if time.time() - startTime >= 30 and time.time() - startTime < 50:
            circleColor = (0,225,0)
        ##Grayscale map
        res_templ = cv.matchTemplate(grayv1, template, cv.TM_CCOEFF_NORMED)
        res_templ = np.clip(res_templ * 255, 0, 255)
        res_templ = res_templ.astype(np.uint8)
        res_templ = cv.resize(res_templ, (1920, 1080))
        

        #Hough transformation
        filter_gray = cv.cvtColor(res_blur, cv.COLOR_BGR2GRAY)
        def drawHough(grayFrame, destFrame, minDist, cannyThreshold, acc, minRad, maxRad):
            circles = cv.HoughCircles(grayFrame, cv.HOUGH_GRADIENT, 1, minDist, param1=cannyThreshold
            , param2=acc, minRadius=minRad, maxRadius=maxRad)
            circleColor = (0,225,0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    cv.circle(destFrame, (i[0], i[1]), i[2], circleColor, 2)
        ##Draw tomato
        def drawHoughTomato(drawlines, grayFrame, destFrame, minDist, cannyThreshold, acc, minRad, maxRad):
            circles = cv.HoughCircles(grayFrame, cv.HOUGH_GRADIENT, 1, minDist, param1=cannyThreshold
            , param2=acc, minRadius=minRad, maxRadius=maxRad)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    y1, y2 = i[1] - round(tomato.shape[0]/2), i[1] + round(tomato.shape[0]/2)
                    x1, x2 = i[0] - round(tomato.shape[0]/2), i[0] + round(tomato.shape[1]/2)
                    if drawlines:
                        points.append((i[0], i[1]))
                        if len(points) > 1:
                            for i in range(len(points)):
                                if i+1 < len(points):
                                    cv.line(destFrame, points[i], points[i+1], (255,0,255), 2)
                    if inRange(x2, y2, x1, y1):
                        alpha_s = tomato[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s

                        for c in range(0, 3):
                            destFrame[y1:y2, x1:x2, c] = (alpha_s * tomato[:, :, c] +
                                                    alpha_l * destFrame[y1:y2, x1:x2, c])

        #Subtitle 
        sub = ""
        #Switch frames
        ##Section 2.1
        if duration < 1  and duration >= 0:
            sub = "BGR"
            showFrame = original
        if duration < 2  and duration >= 1:
            sub = "Gray"
            showFrame = gray
        if duration < 3  and duration >= 2:
            sub = "BGR"
            showFrame = original
        if duration < 4  and duration >= 3:
            sub = "gray"
            showFrame = gray
        ##Section 2.2
        if duration < 7  and duration >= 4:
            sub = "Gaussian 3x3 kernel"
            showFrame = blur
        if duration < 10  and duration >= 7:
            sub = "Gaussian 9x9 kernel"
            showFrame = blurBig
        if duration < 12  and duration >= 10:
            sub =  "Gaussian 15x15 kernel"
            showFrame = blurHuge
        ##Section 3.3
        if duration < 14  and duration >= 12:
            sub =  "Grab object in RGB"
            showFrame = cv.cvtColor(mask_bgr_inv, cv.COLOR_GRAY2BGR)
        if duration < 17  and duration >= 14:
            sub =  "Grab object in HSV"
            showFrame = cv.cvtColor(mask_hsv, cv.COLOR_GRAY2BGR)
        if duration < 20  and duration >= 17:
            sub =  "Grab object in HSV diluted with 7 iterations"
            showFrame = cv.cvtColor(hsv_dilute, cv.COLOR_GRAY2BGR)
        ##Section 3.1
        if duration < 21  and duration >= 20:
            sub =  "Sobel x with kernelsize 7"
            showFrame = frame
            cv.drawContours(showFrame, sobelContour(gray_blur, 7, "x"), -1, (0,255,0), 3)
        if duration < 23  and duration >= 21:
            sub =  "Sobel x with kernelsize 5"
            showFrame = frame
            cv.drawContours(showFrame, sobelContour(gray_blur, 5, "x"), -1, (0,255,255), 3)
        if duration < 24  and duration >= 23:
            sub =  "Sobel y with kernelsize 7"
            showFrame = frame
            cv.drawContours(showFrame, sobelContour(gray_blur, 7, "y"), -1, (0,255,0), 3)
        if duration < 26  and duration >= 24:
            sub =  "Sobel y with kernelsize 5"
            showFrame = frame
            cv.drawContours(showFrame, sobelContour(gray_blur, 5, "y"), -1, (0,255,255), 3)
        ##Section 3.2
        if duration < 28  and duration >= 26:
            sub =  "Houghcirlces with minimum distance 5, threshold 10, accumulator 20, minimum radius 0 and maximumradius 0"
            showFrame = frame
            drawHough(filter_gray, frame, 5, 10, 20, 0, 0)
        if duration < 31  and duration >= 28:
            sub =  "Houghcirlces with minimum distance 5, threshold 50, accumulator 20, minimum radius 0 and maximumradius 0"
            showFrame = frame
            drawHough(filter_gray, frame, 5, 50, 20, 0, 0)
        if duration < 32  and duration >= 31:
            sub =  "Houghcirlces with minimum distance 5, threshold 50, accumulator 20, minimum radius 50 and maximumradius 0"
            showFrame = frame
            drawHough(filter_gray, frame, 5, 50, 10, 50, 0)
        if duration < 34  and duration >= 32:
            sub =  "Houghcirlces with minimum distance 200, threshold 50, accumulator 20, minimum radius 50 and maximumradius 0"
            showFrame = frame
            drawHough(filter_gray, frame, 200, 50, 10, 50, 0)
        if duration < 36  and duration >= 34:
            sub =  "Houghcirlces with minimum distance 400, threshold 50, accumulator 10, minimum radius 50 and maximumradius 200"
            showFrame = frame
            drawHough(filter_gray, frame, 400, 50, 10, 50, 200)
        ##Section 3.3
        if duration < 38  and duration >= 36:
            sub =  "location"
            showFrame = frame
            circles = cv.HoughCircles(filter_gray, cv.HOUGH_GRADIENT, 1, 400, param1=50
            , param2=10, minRadius=50, maxRadius=300)
            circleColor = (0,225,0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    cv.rectangle(frame, (i[0]-i[2], i[1]-i[2]), (i[0]+i[2], i[1]+i[2]), (0,255,255), 2)
        if duration < 41  and duration >= 38:
            sub =  "Grayscale map"
            showFrame = cv.cvtColor(res_templ, cv.COLOR_GRAY2BGR)
        ##Section 4.1
        if duration < 50  and duration >= 41:
            sub =  "Tomato time"
            showFrame = frame
            drawHoughTomato(False, filter_gray, frame, 400, 50, 5, 50, 200)
        if duration >= 50:
            sub =  "Track tomato time"
            showFrame = frame
            drawHoughTomato(True, filter_gray, frame, 400, 50, 5, 50, 200)
        

        cv.putText(showFrame, sub, (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)     
        cv.imshow('test', showFrame) 
        out.write(showFrame)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
mov.release()
cv.destroyAllWindows()




