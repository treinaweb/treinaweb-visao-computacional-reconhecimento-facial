from imutils import face_utils
import dlib
import cv2

shape = "util/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
preditor = dlib.shape_predictor(shape)

cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(grayImage, 0)

    for (i, rect) in enumerate(rects):
        shape_p = preditor(grayImage, rect)
        shape_p = face_utils.shape_to_np(shape_p)

        for (x, y) in shape_p:
            cv2.circle(image, (x,y), 2, (0, 255, 0), -1)

    cv2.imshow("Detector Landmark", image)

    k = cv2.waitKey(5)

    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
