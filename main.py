import  os
import cv2
import numpy as np
import time
import sys
import importlib.util

from email import sendEmailfast
from sms import sendsms

min_conf_threshold = 0.5
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

#Assigning variable to store the model, graph , labels name
MODEL_NAME = 'model'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'

CWD_PATH = os.getcwd()# get current working directory

# importing the labels on which our model ssdmobilenet is trained
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':         #deleting first label if '???' is present
    del(labels[0])
#print

# importing the tflite model file which contains the weights
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()  #allocating all tensors to tflite interpreter

# detecting the dimensions for the input image, to convert our image to same dimension
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

last_epoch = 0
email_update_interval=50000
#starting camera to record video
video = cv2.VideoCapture(0)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

# looping images to get detection video
while True:

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
        print('Reached the end of the video!')
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

            # Draw label
            object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),cv2.FILLED)  # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),2)  # Draw label text
            # check if person is detected in frame
            if (object_name == 'person'):
                if (time.time() - last_epoch) > email_update_interval: #if present time is greater than email time interval then send email and sms
                    # warnings.filterwarnings('ignore')
                    cv2.imwrite('frame.jpg', frame)
                    sendEmailfast(frame)
                    sendsms()
                    last_epoch = time.time()
                    print('Email sent..')

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow(frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

#print(object_name)

# Clean up
video.release()
cv2.destroyAllWindows()