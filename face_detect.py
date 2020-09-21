import dlib
import cv2

def detect_face(img_path):
    shape_predictor_type = "shape_predictor_68_face_landmarks.dat"

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_type)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img_path)
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    face_rects = detector(gray, 1)

    return face_rects

def detect_draw_face(img_path):
    face_rects = detect_face(img_path)
    for i, rect in enumerate(face_rects):
        

if __name__ == "__main__":
    img_path = "images/test02.jpg"
    face_rects = detect_face(img_path)