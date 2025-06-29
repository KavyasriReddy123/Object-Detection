import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from random import randint

model = tf.saved_model.load("ssd_mobilenet_model/saved_model")
image = Image.open("detect.jpg")
image_np = np.array(image)
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
detections = model(input_tensor)
boxes = detections['detection_boxes'].numpy()
classes = detections['detection_classes'].numpy().astype(int)
scores = detections['detection_scores'].numpy()
labels = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
          'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
          'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
          'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
          'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
          'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
          'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
          'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
          'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
          'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
h, w, _ = image_np.shape
for i in range(classes.shape[1]):
    class_id = int(classes[0, i])
    score = scores[0, i]
    if score > 0.5:
        ymin, xmin, ymax, xmax = boxes[0, i]
        xmin, xmax = int(xmin * w), int(xmax * w)
        ymin, ymax = int(ymin * h), int(ymax * h)
        class_name = labels[class_id]
        color = (randint(0, 256), randint(0, 256), randint(0, 256))
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image_np, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
plt.imshow(image_np)
plt.axis('off')
plt.show()
