"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
#....................................................................................

def extract_face(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    # print x , y, w ,h
    horizontal_offset = np.int((offset_coefficients[0] * w))
    vertical_offset = np.int((offset_coefficients[1] * h))

    extracted_face = gray[y + vertical_offset:y + h,
                      x + horizontal_offset:x - horizontal_offset + w]
    # print extracted_face.shape
    new_extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0],
                                                48. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(np.float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face



from scipy.ndimage import zoom
def detect_face(frame):
        cascPath = "./models/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(48, 48), flags = cv2.CASCADE_SCALE_IMAGE)
        return gray, detected_faces

import cv2
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*"MP4V")
out = cv2.VideoWriter('./video/v', fourcc, -1, 20.0, (1920,1080))
while True:
    # Capture frame-by-frame
    #sleep(4)
    ret, frame= video_capture.read()

    # detect faces
    gray1, detected_faces = detect_face(frame)

    face_index = 0
    for rect in detected_faces:
        (x, y, w, h) = rect
        if w>100:
            frame1 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            extracted_face = extract_face(gray1, rect, (0.075, 0.05))
            frame1[face_index * 48: (face_index + 1) * 48, -49:-1, :] = cv2.cvtColor(extracted_face * 255,
                                                                                    cv2.COLOR_GRAY2RGB)

            raw_img = frame
            gray = rgb2gray(raw_img)
            gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

            img = gray[:, :, np.newaxis]

            img = np.concatenate((img, img, img), axis=2)
            img = Image.fromarray(img)
            inputs = transform_test(img)

            class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

            net = VGG('VGG19')
            checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'),map_location='cpu')
            net.load_state_dict(checkpoint['net'])
            net.cpu()
            net.eval()

            ncrops, c, h, w = np.shape(inputs)

            inputs = inputs.view(-1, c, h, w)
            inputs = inputs.cpu()
            inputs = Variable(inputs, volatile=True)
            outputs = net(inputs)

            outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

            score = F.softmax(outputs_avg)
            _, predicted = torch.max(outputs_avg.data, 0)

            cv2.putText(frame, str(class_names[int(predicted.cpu().numpy())]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), lineType=cv2.LINE_AA)
            face_index += 1
            #print(str(class_names[int(predicted.cpu().numpy())]))

    #io.imshow(raw_img)
    #io.show()
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break






out.write(frame)

video_capture.release()
cv2.destroyAllWindows()








