!pip install mediapipe
import cv2
from google.colab.patches import cv2_imshow
import mediapipe as mp
# INITIALIZING OBJECTS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    image = cv2.imread("/content/face.jpeg", cv2.IMREAD_COLOR)
    # image = resize_image(image, 20)

    # Flip the image horizontally and convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Detect the face landmarks
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert back to the BGR color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the face mesh annotations on the image.
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())

    # Display the image
    cv2_imshow(image)

!git clone https://github.com/vardanagarwal/Proctoring-AI
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from google.colab.patches import cv2_imshow
def get_face_detector(modelFile = "/content/Proctoring-AI/face_detection/models/res10_300x300_ssd_iter_140000.caffemodel",
                      configFile = "/content/Proctoring-AI/face_detection/models/deploy.prototxt"):
    """
    Get the face detection caffe model of OpenCV's DNN module
    
    Parameters
    ----------
    modelFile : string, optional
        Path to model file. The default is "models/res10_300x300_ssd_iter_140000.caffemodel".
    configFile : string, optional
        Path to config file. The default is "models/deploy.prototxt".

    Returns
    -------
    model : dnn_Net

    """
    modelFile = "/content/Proctoring-AI/face_detection/models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "/content/Proctoring-AI/face_detection/models/deploy.prototxt"
    model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def get_landmark_model(saved_model='/content/Proctoring-AI/models/pose_model'):
    """
    Get the facial landmark model. 
    Original repository: https://github.com/yinguobing/cnn-facial-landmark

    Parameters
    ----------
    saved_model : string, optional
        Path to facial landmarks model. The default is 'models/pose_model'.

    Returns
    -------
    model : Tensorflow model
        Facial landmarks model

    """
    model = tf.saved_model.load(saved_model)
    # model = keras.models.load_model(saved_model)
    return model

def find_faces(img, model):
    """
    Find the faces in an image
    
    Parameters
    ----------
    img : np.uint8
        Image to find faces from
    model : dnn_Net
        Face detection model

    Returns
    -------
    faces : list
        List of coordinates of the faces detected in the image

    """
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces


def get_square_box(box):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        return box
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]

def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]

def detect_marks(img, model, face):
    """
    Find the facial landmarks in an image from the faces

    Parameters
    ----------
    img : np.uint8
        The image in which landmarks are to be found
    model : Tensorflow model
        Loaded facial landmark model
    face : list
        Face coordinates (x, y, x1, y1) in which the landmarks are to be found

    Returns
    -------
    marks : numpy array
        facial landmark points

    """

    offset_y = int(abs((face[3] - face[1]) * 0.1))
    box_moved = move_box(face, [0, offset_y])
    facebox = get_square_box(box_moved)
    
    face_img = img[facebox[1]: facebox[3],
                     facebox[0]: facebox[2]]
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # # Actual detection.
    predictions = model.signatures["predict"](
        tf.constant([face_img], dtype=tf.uint8))

    # Convert predictions to landmarks.
    marks = np.array(predictions['output']).flatten()[:136]
    marks = np.reshape(marks, (-1, 2))
    
    marks *= (facebox[2] - facebox[0])
    marks[:, 0] += facebox[0]
    marks[:, 1] += facebox[1]
    marks = marks.astype(np.uint)

    return marks

def draw_marks(image, marks, color=(0, 0, 255)):
    """
    Draw the facial landmarks on an image

    Parameters
    ----------
    image : np.uint8
        Image on which landmarks are to be drawn.
    marks : list or numpy array
        Facial landmark points
    color : tuple, optional
        Color to which landmarks are to be drawn with. The default is (0, 255, 0).

    Returns
    -------
    None.

    """
    for mark in marks:
        cv2.circle(image, (mark[0], mark[1]), 5, color, -1, cv2.LINE_AA)

def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

face_model = get_face_detector()
landmark_model = get_landmark_model()
from google.colab import drive

drive.mount('/content/gdrive')
def let_do_it(img, output):
    rects = find_faces(img, face_model)

    for rect in rects:
        marks = detect_marks(img, landmark_model, rect)
        draw_marks(img, marks)

    cv2.imwrite(output, img)
folder_path = "/content/gdrive/MyDrive/Raw_100"
output_path = "/content/gdrive/MyDrive/output"

for filename in os.listdir(folder_path):
    img = cv2.imread(os.path.join(folder_path,filename))

    res_img_path = os.path.join(output_path, filename)
    let_do_it(img, res_img_path)
    print(res_img_path