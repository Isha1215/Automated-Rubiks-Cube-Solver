import cv2
import numpy as np
import utilities
kernel=np.ones((5,5),np.uint8)
def empty(a):
    pass

def getcontours(img,img_contour):
    contours,heirarchy=cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        peri=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.02*peri,True)
        if area>1000 and area<2000:
            if len(approx)<9:
                cv2.drawContours(img_contour,cnt,-1,(0,255,255),2)

cv2.namedWindow("T")
cv2.resizeWindow("T",640,240)
cv2.createTrackbar("hue_min","T",32,179,empty)
cv2.createTrackbar("hue_max","T",179,179,empty)
cv2.createTrackbar("sat_min","T",0,255,empty)
cv2.createTrackbar("sat_max","T",255,255,empty)
cv2.createTrackbar("val_min","T",58,255,empty)
cv2.createTrackbar("val_max","T",255,255,empty)
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars",640,240)
cv2.createTrackbar("Threshold1","Trackbars",150,255,empty)
cv2.createTrackbar("Threshold2","Trackbars",255,255,empty)
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while True:
    success,img=cap.read()
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hue_min=cv2.getTrackbarPos("hue_min","T")
    hue_max=cv2.getTrackbarPos("hue_max","T")
    sat_min=cv2.getTrackbarPos("sat_min","T")
    sat_max=cv2.getTrackbarPos("sat_max","T")
    val_min=cv2.getTrackbarPos("val_min","T")
    val_max=cv2.getTrackbarPos("val_max","T")
    lower=np.array([hue_min,sat_min,val_min])
    upper=np.array([hue_max,sat_max,val_max])
    mask=cv2.inRange(img_hsv,lower,upper)
    #cv2.imshow("mask",mask)
    final=cv2.bitwise_and(img,img,mask=mask)
    #cv2.imshow("f1",img)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    img_canny = cv2.Canny(img_gray, threshold1,threshold2)
    img_dilation = cv2.dilate(img_canny, kernel, iterations=1)
    getcontours(img_dilation, final)
    f=utilities.StackedImages(0.6,([img,img_hsv],[mask,final]))
    cv2.imshow("f",f)
    if cv2.waitKey(1000) & 0xFF==ord('p'):
        break