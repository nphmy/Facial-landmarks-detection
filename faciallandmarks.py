# import library
import cv2 
from google.colab.patches import cv2_imshow
import dlib
import numpy as np
# Create 'model' folder
!mkdir model
!mkdir images
!wget https://i.pinimg.com/564x/46/fc/32/46fc3210874f36a0fa714a224390dd9c.jpg -O "/content/images/lisa.jpg"
# Download pre-trained model from github
!wget https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat -P '/content/model'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/content/model/shape_predictor_68_face_landmarks.dat')
def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized
demo = cv2.imread("/content/_DSC0782.JPG", cv2.IMREAD_COLOR)
image = resize_image(demo, 40)
width, height, channels = image.shape
print(image.shape)

# Convert the image color to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect the face
rects = detector(gray, 1)
# Detect landmarks for each face
for rect in rects:
    # Get the landmark points
    shape = predictor(gray, rect)
# Convert it to the NumPy Array
    shape_np = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    shape = shape_np

    # Display the landmarks
    for i, (x, y) in enumerate(shape):
    # Draw the circle to mark the keypoint 
        cv2.circle(image, (x, y), round(width * height / 500000), (0, 0, 255), -1)
    
# Display the image
cv2_imshow(image)