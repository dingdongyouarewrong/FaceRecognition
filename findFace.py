import cv2, numpy
from keras.preprocessing.image import img_to_array

from model import create_model, findCosineDistance
haar_file = 'cascade/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

vector = 0
webcam = cv2.VideoCapture(0)

count_of_faces = 1
model = create_model()

img1_descriptor = numpy.load("descriptors/img1_descriptor.npy")
img2_descriptor = numpy.load("descriptors/img2_descriptor.npy")

def get_name_from_base(descriptors):
    if findCosineDistance(descriptors, img1_descriptor) <0.3:
        return "1"
    elif findCosineDistance(descriptors, img2_descriptor) <0.3:
        return "2"

def mat_preprocessing(detected_face):
    image_pixels = img_to_array(detected_face)
    image_pixels = numpy.expand_dims(image_pixels, axis = 0)
    image_pixels /= 127.5
    image_pixels -=1
    return image_pixels

while count_of_faces<30:
    (_, image) = webcam.read()
    faces = face_cascade.detectMultiScale(image,1.3,5)

    for (x,y,width,height) in faces:
        if width>130:
            detected_face = image[int(y):int(y+height), int(x):int(x+width)]
            detected_face = cv2.resize(detected_face,(224,224))
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 3)

            image_pixels = mat_preprocessing(detected_face)

            captured_representation = model.predict(image_pixels)[0,:]
            name = get_name_from_base(captured_representation)

            cv2.putText(image, name, (x, y-10), cv2.QT_FONT_NORMAL, 0.7,(0,255,0),2)

    cv2.imshow('FaceRecognition',image)

    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break

webcam.release()
