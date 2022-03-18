import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt


def plot_live():
    fig = plt.figure()
    plt.plot(timing, current_visitor)
    plt.xticks(np.arange(0, int(len(timing) / 20), 10), rotation='vertical')
    plt.subplots_adjust(bottom=0.20)
    fig.canvas.draw()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plt.close(fig)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('im', image)


def info_live(curr, time_c):
    img_1 = np.zeros([200, 400, 1], dtype=np.uint8)
    img_1.fill(255)
    index_max = current_visitor.index(max(current_visitor))
    cv2.putText(img_1, 'High time : ' + str(current_visitor[index_max]), (10, 25), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img_1, 'time : ' + str(timing[index_max]), (200, 25), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img_1, 'Current visitors : ' + str(curr), (10, 50), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img_1, 'time : ' + str(time_c), (200, 50), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('info', img_1)


def filtering(original_frame, background):
    # kernel_Op and kernel_Cl are 2D matrix we use som parameter til morphologyEx function
    kernel_Op = np.ones((4, 4), np.uint8)
    kernel_Cl = np.ones((15, 15), np.uint8)
    # for each frame vi use our fg_bg on it to to sperate objects from background
    fg_mask = background.apply(original_frame)
    cv2.imshow("mask_raw", fg_mask)
    # the cv2.threshold and parameter cv2.THRESH_BINARY return a frame with min/max color base on intensity of the pixel
    _, imBin = cv2.threshold(fg_mask, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("shadow", imBin)
    # the cv2.morphologyEx with parameter cv2.MORPH_OPEN return a frame without those pixels away from objects
    mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernel_Op)
    cv2.imshow("mask_op", mask)
    # the cv2.morphologyEx with parameter cv2.MORPH_CLOSE return a frame after its full hols in objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_Cl)
    cv2.imshow("mask_cl", mask)
    return mask


class Person:
    def __init__(self, xi, yi, d):
        self.x = xi
        self.y = yi
        self.dir = d
        self.state = False


cap = cv2.VideoCapture("testing_video.mp4")
count_up, count_down = 0, 0
width = int(cap.get(3))
height = int(cap.get(4))
mid = int(height / 2)
# areaTH is areal to the part of frame that we use later
# 150 is a constant
person_area_min = (height * width) / 110
person_area_max = (height * width) / 22
line_up = int(1 * (height / 6))
line_down = int(4 * (height / 6))

# font is a text type
font = cv2.FONT_HERSHEY_SIMPLEX
# createBackgroundSubtractorMOG2 is a method to seperate the objects from background
# and its return the objects with white and background with black color
fg_bg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
persons = []
timing = [0]
current_visitor = [0]
# in the next while loop all code is happening for each frame
while cap.isOpened():
    _, frame = cap.read()
    mask = filtering(frame, fg_bg)
    # findContours function find all objects in a frame
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        persons.clear()
    for cnt in contours:
        # contourArea find the areal of the box around objects
        area = cv2.contourArea(cnt)
        if person_area_min < area < person_area_max:
            # cv2.moments return a dic with data about objects
            M = cv2.moments(cnt)
            # next to line give x,y coordinate to masse center of objects
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)
            new = True
            if cy in range(0, height):
                for i in persons:
                    # next line is a way to check if the person(object) is a new one or a person we already detected
                    if abs(cx - i.x) <= int(w / 3) and abs(cy - i.y) <= int(h / 3):
                        new = False
                        i.x, i.y = cx, cy
                        if i.state:
                            continue
                        else:
                            if i.dir == 'up':
                                if i.y <= line_up:
                                    count_up += 1
                                    i.state = True
                            elif i.dir == 'down':
                                if i.y >= line_down:
                                    count_down += 1
                                    i.state = True
                            else:
                                break
                if new:
                    # giv a direction base on first detection coordinate
                    direc = ''
                    if cy < mid:
                        direc = 'down'
                    if cy > mid:
                        direc = 'up'
                    per = Person(cx, cy, direc)
                    persons.append(per)

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    str_up = 'UP: ' + str(count_up)
    str_down = 'DOWN: ' + str(count_down)
    tid = datetime.datetime.now()
    tid_c = tid.strftime('%H:%M:%S')
    timing.append(tid_c)
    cur = count_down - count_up
    current_visitor.append(cur)
    frame = cv2.line(frame, (0, line_down), (width, line_down), color=(255, 0, 0), thickness=2)
    frame = cv2.line(frame, (0, line_up), (width, line_up), color=(0, 0, 255), thickness=2)
    cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('Frame', frame)
    #info_live(cur, tid_c)
    key = cv2.waitKeyEx(30)
    if key == ord('p'):
        plot_live()
    if key == ord('0'):
        break
cap.release()
cv2.destroyAllWindows()
