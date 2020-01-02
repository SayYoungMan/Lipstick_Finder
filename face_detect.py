from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt

def rgb2hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def crop_poly(image,points):
    #crop the rectangle
    rect = cv2.boundingRect(points)
    x,y,w,h = rect
    cropped = image[y:y+h, x:x+w].copy()

    #Make mask
    points = points - points.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255,255,255), -1, cv2.LINE_AA)

    #Do bit-op
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)

    #add white background
    bg = np.ones_like(cropped, np.uint8)*255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst = bg + dst
    return dst

def find_lips(image, shape):
    # Make Separate Array of Upper and Lower lips
    upper_lip = np.concatenate((shape[48:55],np.flipud(shape[60:65])))
    lower_lip = np.concatenate((shape[54:60],shape[48:49],shape[60:61],np.flipud(shape[65:68])))

    #Draw the polygon of lips
    cv2.polylines(image,[upper_lip],True,(255,255,255))
    cv2.polylines(image,[lower_lip],True,(255,255,255))

    #Crop the polygons
    upperlip_img = crop_poly(image, upper_lip)
    lowerlip_img = crop_poly(image, lower_lip)
    return upperlip_img, lowerlip_img

def get_colors(image, no_color):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)
    clf = KMeans(n_clusters = no_color)
    labels = clf.fit_predict(modified_image)
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    center_colors = clf.cluster_centers_

    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb2hex(ordered_colors[i]) for i in counts.keys()]
    #rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if '#ffffff' in hex_colors:
        counts.pop(hex_colors.index('#ffffff'))
        hex_colors.remove('#ffffff')
    elif '#fefeff' in hex_colors:
        counts.pop(hex_colors.index('#fefeff'))
        hex_colors.remove('#fefeff')
    
    return hex_colors, counts

shape_predictor_type = "shape_predictor_68_face_landmarks.dat"
#img_path = "images/test01.png"
img_path = "images/test02.jpg"

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_type)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(img_path)
#image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
 
    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    upperlip_img, lowerlip_img = find_lips(image, shape)
    upperlip_color, upperlip_counts = get_colors(upperlip_img, 8)
    lowerlip_color, lowerlip_counts = get_colors(lowerlip_img, 8)
    upperlip_analysis = dict(zip(upperlip_color, upperlip_counts.values()))
    lowerlip_analysis = dict(zip(lowerlip_color, lowerlip_counts.values()))
    lip_analysis = dict(Counter(upperlip_analysis) + Counter(lowerlip_analysis))
    plt.figure(figsize = (8, 6))
    plt.pie(lip_analysis.values(), labels = lip_analysis.keys(), colors = lip_analysis.keys())
    plt.show()

# show the output image with the face detections + facial landmarks
#cv2.imshow("Output", image)
#cv2.waitKey(0)