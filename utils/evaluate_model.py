import cv2
import glob
import sys
import numpy as np
import tensorflow as tf
import time
import json
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from progressbar import progressbar
import os
sys.path.append('src/')
from model import create_model, yolo_eval, box_iou
from utils import get_classes, get_anchors, letterbox_image

# This might need params or configs for max_boxes, score_thresh, and iou_thresh
max_boxes = 100
score_thresh = 0.5
iou_thresh = 0.90

# Read configuration file
config_file = f"val/{sys.argv[1]}/{sys.argv[1]}_val_config"

'''
The configuration file should contain the following information in the order specified below:
- Image directory path
- Classes file path
- Anchors file path
- YOLO weights file path
- YOCO weights file path
- Ground truth file path
- YOLO prediction file path
- YOCO prediction file path
'''
with open(config_file, 'r') as f:
    img_dir_path = f.readline().rstrip('\n')    # Path to the directory containing images
    classes_path = f.readline().rstrip('\n')    # Path to the file containing class names
    anchors_path = f.readline().rstrip('\n')    # Path to the file containing anchor box information
    yolo_weights_path = f.readline().rstrip('\n')    # Path to the YOLO weights file
    yoco_weights_path = f.readline().rstrip('\n')    # Path to the YOCO weights file
    gt_file = f.readline().rstrip('\n')    # Path to the ground truth file
    yolo_pred_file = f.readline().rstrip('\n')    # Path to the YOLO prediction file
    yoco_pred_file = f.readline().rstrip('\n')    # Path to the YOCO prediction file


weights_path = yolo_weights_path if len(sys.argv) > 2 and sys.argv[2].upper() == "YOLO" else yoco_weights_path
pred_file = yolo_pred_file if len(sys.argv) > 2 and sys.argv[2].upper() == "YOLO" else yoco_pred_file
generate_new_prediction = len(sys.argv) > 3 or not os.path.isfile(pred_file)

img_paths = glob.glob(img_dir_path + "/*")

class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
input_shape = (416, 416)

if generate_new_prediction:
    batch_size = 16
    model = create_model('inf', batch_size, input_shape, anchors, num_classes, weights_path=weights_path, freeze_body=0)

    categories = [{"id": c, "name": class_names[c], "supercategory": "none"} for c in range(num_classes)]

    pred_json = []
    ann_id = 0
    pred_id = 0

    for path in progressbar(img_paths):
        img = cv2.imread(path)
        img_id = os.path.splitext(os.path.basename(path))[0]

        if img is None:
            print('Wrong path:', path)
            exit()
        else:
            img_size = 416
            height, width, channels = img.shape
            img = cv2.resize(img, input_shape)
            image_data = np.divide(img, 255., casting="unsafe")
            image_data = np.expand_dims(image_data, 0)

            start = time.time()
            outputs = model.predict(image_data, verbose=0)
            end = time.time()
            out_boxes, out_scores, out_classes = yolo_eval(outputs, anchors, num_classes, input_shape, max_boxes, score_thresh, iou_thresh)

            print('Detected {} objects in {}s ({})'.format(len(out_boxes), (end-start), path))

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)

                top, left, bottom, right = box
                height = bottom - top
                width = right - left

                bounding_box = [int(left), int(top), int(width), int(height)]
                pred_json.append({
                    "image_id": int(img_id),
                    "category_id": int(c),
                    "bbox": bounding_box,
                    "score": float(score)
                })
                pred_id += 1

    with open(pred_file, 'w') as f:
        json.dump(pred_json, f, indent=4)

coco_anno = COCO(annotation_file=gt_file)
coco_pred = coco_anno.loadRes(pred_file)
coco_eval = COCOeval(coco_anno, coco_pred, "bbox")

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()