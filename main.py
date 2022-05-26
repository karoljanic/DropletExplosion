import numpy as np
import cv2
import matplotlib.pyplot as plt

excel_file = open('results/data.txt', 'w')


# function calculating distance between points: (x1, y1) and (x2, y2)
def distance(x1, y1, x2, y2):
    return (((x1 - x2) * (x1 - x2)) + ((y1 - y2) * (y1 - y2))) ** .5


# function checking if circle with center (x1, y1) and radius r1
# is included in circle with center (x2, y2) and radius r2
def circle_in_circle(x1, y1, x2, y2, r1, r2):
    if distance(x1, y1, x2, y2) + r2 <= r1:
        return True
    else:
        return False


# finding number-th circle in terms of size in the picture img
# it returns tuple with tuple with circle center coords and radius
def find_circle(img, number):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_and_white_image = cv2.fastNlMeansDenoising(gray_image, None, 20, 7, 21)

    circles = cv2.HoughCircles(black_and_white_image, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles = np.round(circles[0, :]).astype("int")
    mc = circles[number]

    mcx = int(mc[0])
    mcy = int(mc[1])
    mcr = int(mc[2])

    return (mcx, mcy), mcr


# finding circle which is dish rim
def find_main_circle(img):
    center, r = find_circle(img, 0)

    return center, int(1 * r)


# constants used in program
frame_rate = 100
max_area = 1000
save = False

# reading video
cap = cv2.VideoCapture('videos/v1.mkv')
cap_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = cap_length % frame_rate

# output data
data = {}
average_of_radiuses = {}

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
_, background = cap.read()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# preparing video to save result of analyse on input video
if save:
    out = cv2.VideoWriter('results/wow_v2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (frame_width, frame_height))

main_circle = find_main_circle(background)


# function to drawing dish rim on output video
def draw_main_circle(img):
    cv2.circle(img, main_circle[0], main_circle[1], (0, 0, 255), 2)


# checking if pixel with index i,j is included in main_circle(rim of dash)
def pixel_in_circle(i, j):
    return (i - main_circle[0][0]) ** 2 + (j - main_circle[0][1]) ** 2 <= main_circle[1] ** 2


# drawing first frame of video with main_circle to check if it loaded properly
draw_main_circle(background)
cv2.imshow('background', background)
cv2.waitKey()

# object which is responsible for deleting background of video
backSub = cv2.createBackgroundSubtractorMOG2()

# analysing frames of video
while current_frame <= cap_length:
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()

    # checking if frame is read correctly
    if not ret:
        break

    # creating copy of frame to do not change input video
    copy_of_frame = frame.copy()

    # deleting background of frame
    copy_of_frame = backSub.apply(copy_of_frame)
    copy_2_of_frame = copy_of_frame.copy()

    # creating and applying filter on frame to delete 'mis-pixels'
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    copy_of_frame = cv2.morphologyEx(copy_of_frame, cv2.MORPH_CLOSE, kernel)

    # finding all contours on frame
    contours, hierarchy = cv2.findContours(copy_of_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    contours_on_image = np.full_like(frame, 255)
    good_circles = []
    # iterating by contours and checking if meet droplet properties
    # if yes: adding it to result array
    for component in zip(contours, hierarchy):
        area = cv2.contourArea(component[0])
        if area < max_area:
            (x, y), radius = cv2.minEnclosingCircle(component[0])
            if circle_in_circle(*main_circle[0], x, y, main_circle[1], radius):
                good_circles.append([int(x), int(y), int(radius)])

    black_screen = np.full_like(background, 0)

    sor = 0
    # calculating sum of radiuses of found droplets
    # and drawing found circles
    for c in good_circles:
        sor += c[2]
        cv2.circle(frame, (int(c[0]), int(c[1])), 2, (0, 255, 0), cv2.FILLED, 2)
        cv2.circle(black_screen, (int(c[0]), int(c[1])), 2, (0, 255, 0), cv2.FILLED, 2)

    # calculating average of radiuses of found droplets
    if not len(good_circles) == 0:
        data.update({current_frame: len(good_circles)})
        ar = sor / len(good_circles)
        average_of_radiuses.update({current_frame: (ar * 150) / main_circle[1]})

    # preparing output frame to result video
    draw_main_circle(frame)
    draw_main_circle(black_screen)
    cv2.imshow('frame', frame)
    if save:
        out.write(black_screen)
    if cv2.waitKey(1) == ord('q'):
        break

    current_frame += frame_rate

# closing everything
if save:
    out.release()
cap.release()
cv2.destroyAllWindows()

# saving data to txt file used in Excel analyse
for k, v1, v2 in zip(data.keys(), data.values(), average_of_radiuses.values()):
    excel_file.write(str(k) + " " + str(v1) + " " + str(v2) + "\n")

# drawing plots to see result
fig, axs = plt.subplots(2)
axs[0].bar(data.keys(), data.values(), width=75)
axs[1].bar(average_of_radiuses.keys(), average_of_radiuses.values(), width=75)

fig.suptitle('liczba w czasie / sredni rozmiar w czasie[mm]')
fig.subplots_adjust(wspace=0, hspace=0)
axs[0].tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off

# saving generated plots
# fig.savefig('results/wykres.png')
fig.show()

# closing file with data to Excel
excel_file.close()
