import os
MODEL_NAME = 'model'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
CWD_PATH = os.getcwd()
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
print(PATH_TO_LABELS)