from cv2 import cv2
import numpy as np
from IPython.display import clear_output

# get location where mouse is clicked on the image
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append([x, y])
        cv2.imshow('grab', add_dots(copy, x, y))

# mark a location on the image
def add_dots(img, x, y):
    s = 2
    iWithRect = cv2.rectangle(copy, (x+s, y+s), (x-s, y-s), (255,255,255), 2) 
    return iWithRect

def create_transformed(baseImg):
    ## transform image and place it onto the second image
    movePoints = [[0,0], [0, newSize[1]], [newSize[0], newSize[1]], [newSize[0], 0]]
    H,_ = cv2.findHomography(np.array(locations), np.array(movePoints))
    # warp and resize
    warped = cv2.warpPerspective(baseImg, H, newSize)
    return warped

# for reshaping after homography
locations = []
newSize = (96,72)

# import training video
cap = cv2.VideoCapture("original_data/train.mp4") # training data video
frameLength = 20400 # for training data
# frameLength = 10798 # for test data

# grab first frame
cap.grab()
ret, frame1 = cap.retrieve()

# copy image and get points to crop from the image
copy = frame1.copy()
cv2.imshow('grab', copy)
cv2.setMouseCallback('grab', get_xy, param=locations)
cv2.waitKey(0)

frame1 = create_transformed(frame1)
cv2.imshow('Area To Use',frame1)
cv2.waitKey(0)

prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

frameCount = 0
allFrames = []
while(frameCount < frameLength-1):
    cap.grab()
    ret, frame2 = cap.retrieve()
    frame2 = create_transformed(frame2)
    nextImg = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, nextImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB_FULL)
    bw = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    prvs = nextImg

    allFrames.append(bw)
    frameCount += 1

    clear_output(wait=True)
    print(frameCount)

allFrames = np.array(allFrames)
np.save('train_data/tain_homography.npy', allFrames)
cap.release()
cv2.destroyAllWindows()
