import numpy as np
import cv2
import kociemba
from operator import  itemgetter

square_side_ratio=1.5
min_square_length=0.08
min_area_ratio=0.7
min_square_size=0.10
max_square_size=0.5
#'b':((102,153,153),(153,255,255))
#'g':((50,100,100),(90,255,240))
#'y':((10, 60,150),(60,255,255))
#'o':((0,96,247),(255,247,255))
color_lims={'w':((0,  0, 90),(255, 80,255)),'g':((50,100,100),(90,255,240)),'o':((0,102,255),(179,204,255)),'b':((90,153,153),(153,255,255)),'y':((35,51,204),(74,153,255)),'r':((0,201,0),(25,250,223))}

def index_cube(pts):
    if len(pts)!=9:
        return None
    pts=[list(pts[i])+[i] for i in range(len(pts))]
    pts.sort(key=itemgetter(1))
    mat=[[pts[3*i+j] for j in range(3)]for i in range (3)]
    for i in range(3):
        mat[i].sort(key=itemgetter(0))
    for i in range(3):
        for j in range(3):
            mat[i][j]=mat[i][j][2]
    return mat

def get_color_mask(im,color_name,isBGR=False):
    if isBGR:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    color_lim=color_lims[color_name]
    """if color_name in 'RO':
        big_hue=max((color_lim[0][0],color_lim[1][0]))
        small_hue = max((color_lim[0][0], color_lim[1][0]))
        lower_red=[(0,color_lim[0][1],color_lim[0][2]),(small_hue,color_lim[1][1],color_lim[1][2])]
        upper_red=[(big_hue,color_lim[0][1],color_lim[0][2]),(180,color_lim[1][1],color_lim[1][2])]
        mask=cv2.inRange(im,upper_red[0],upper_red[1])|cv2.inRange(im,lower_red[0],lower_red[1])
    else:"""
    mask = cv2.inRange(im, color_lim[0], color_lim[1])
    return mask
def detect_square(im,color_name):
    def remove_bad_contours(contour):
        new_conts=[]
        for cont in contour:
            bound_rect=cv2.minAreaRect(cont)
            length=float(bound_rect[1][0])
            breadth=float(bound_rect[1][1])
            try:
                if max(length/breadth,breadth/length)>square_side_ratio:
                    continue
                if cv2.contourArea(cont)/(length*breadth)<min_area_ratio:
                    continue
                if not max_square_size*im.shape[0]>max(length,breadth)>min_square_size*im.shape[0]:
                    continue
                new_conts.append(cont)
            except ZeroDivisionError:
                continue
        return new_conts

    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask=get_color_mask(im,color_name)
    contour,heirarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour=remove_bad_contours(contour)
    return [cv2.minAreaRect(cont) for cont in contour]
def sol(input1):
    input2=[[[]],[[]],[[]],[[]],[[]],[[]]]
    for i in range(6):
        if (input1[i][1][1] == 'w'):
            input2[0] = input1[i]
            break

    for i in range(6):
        if (input1[i][1][1] == 'r'):
            input2[1] = input1[i]
            break

    for i in range(6):
        if (input1[i][1][1] == 'g'):
            input2[2] = input1[i]
            break

    for i in range(6):
        if (input1[i][1][1] == 'y'):
            input2[3] = input1[i]
            break

    for i in range(6):
        if (input1[i][1][1] == 'o'):
            input2[4] = input1[i]
            break

    for i in range(6):
        if (input1[i][1][1] == 'b'):
            input2[5] = input1[i]
            break
    print(input2)
    for i in range(0,6):
        for j in range(0,3):
            for k in range(0,3):
                if input2[i][j][k]=='w':
                    input2[i][j][k]='U'
                elif input2[i][j][k]=='y':
                    input2[i][j][k] = 'D'
                elif input2[i][j][k]=='r':
                    input2[i][j][k] = 'R'
                elif input2[i][j][k]=='o':
                    input2[i][j][k] = 'L'
                elif input2[i][j][k]=='g':
                    input2[i][j][k] = 'F'
                elif input2[i][j][k]=='b':
                    input2[i][j][k] = 'B'
    b=''
    for i in range(0,6):
        for j in range(0,3):
            for k in range(0,3):
                b+=input2[i][j][k]
    print(b)
    a=kociemba.solve(b)
    print(a)
def get_cube_state(im):
    color_detected=[]
    cube_state=[[None]*3 for i in range(3)]
    for color,color_lim in zip(color_lims.keys(),color_lims.values()):
        rects=detect_square(im,color)
        for rect in rects:
            color_detected.append((color, rect))
    index_mat=index_cube([p[1][0]for p in color_detected])
    if index_mat!=None:
        for i in range (3):
            for j in range(3):
                cube_state[i][j] = color_detected[index_mat[i][j]][0]

        return cube_state
    else:
        return None
def draw_rects(im,rects,color,iscolHSV=True):
    if iscolHSV:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    boxpts=np.array([cv2.boxPoints(rect) for rect in rects],dtype=np.int32)
    cv2.polylines(im,boxpts,True,color,thickness=5)
    return cv2.cvtColor(im,cv2.COLOR_HSV2BGR)
def get_color(char):
    color = {'r': (255, 0, 0), 'o': (255, 128, 0), 'b': (0, 0, 255),
         'g': (0, 255, 0), 'w': (255, 255, 255), 'y': (255, 255, 0)}
    return tuple(list(color[char.lower()])[::-1])
def draw_cube(im,cube_state):
    for i in range(3):
        for j in range(3):
            if cube_state!=None:
                im=cv2.rectangle(im,(10+40*j,10+40*i),(40+40*j,40+40*i),get_color(cube_state[i][j]),3)
                im=cv2.putText(im,cube_state[i][j],(17+40*j,32+40*i),cv2.FONT_HERSHEY_COMPLEX,0.7,get_color(cube_state[i][j]),1)
            else:
                im = cv2.rectangle(im, (10 + 40 * j, 10 + 40 * i), (40 + 40 * j, 40 + 40 * i),
                                   (128,128,128), 3)
                im = cv2.putText(im,'?', (17 + 40 * j, 32 + 40 * i), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                                 (128,128,128), 1)
    return im
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3,640)
cap.set(4,480)
cube=[]
count=0
list1=["Start","Top-Face","Bottom-Face","Right-Face","Left-Face","Front-Face","Back-Face"]
while True:
    success,im=cap.read()
    disp_im=np.array(im)
    for color in color_lims:
        sq=detect_square(im,color)
        disp_im=draw_rects(disp_im,sq,get_color(color))
    disp_im = cv2.putText(disp_im, list1[count], (10, 180), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0),4)
    cube_state=get_cube_state(im)
    disp_im=draw_cube(disp_im,cube_state)
    cv2.imshow("a", disp_im)
    k=cv2.waitKey(1)
    if k==ord('q') or k==ord('p'):
        break
    if k==ord('n'):
        count+=1
    if k==ord('s'):
        if get_cube_state(im) == None:
            cube = []
            count = 0
        else:
            cube.append(get_cube_state(im))
print(cube)
sol(cube)





