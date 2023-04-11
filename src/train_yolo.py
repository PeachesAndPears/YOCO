"""
Retrain the YOLO model for your own dataset.
"""

''' Only use a certain GPU number '''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from model import preprocess_true_boxes, yolo_body, yolo_loss, create_model
from utils import get_random_yolo_data, get_random_domain_data
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)


def main():
    model_name = 'YOLOv3-blendBennu'
    log_dir = '/mnt/fastdata0/tbchase/training_logs/' + model_name  + '/'
    
    # Training data/variable setup
    box_anno_path = "../model_data/annotations/box_blendBennu.txt"
    classes_path  = '../model_data/class_lists/blendBennu_classes.txt'
    anchors_path  = '../model_data/anchors/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    input_shape = (416,416) # multiple of 32, hw
    
    # Training/loss parameters
    batch_size  = 32
    start_epoch = 0
    # ----------------
    stage1_epochs = 25
    stage2_epochs = 25
    stage1_lr = 1e-3
    stage2_lr = 1e-4

    # Create YOLOv3 model
    model = create_model('yolo', batch_size, input_shape, anchors, num_classes,
        weights_path='../model_data/saved_weights/yolo.h5') # make sure you know what you freeze
    model.summary()

    # Callbacks
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(filepath=log_dir + model_name + '_ep{epoch:03d}.ckpt', monitor='yolo_loss', save_weights_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='yolo_loss', factor=0.1, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='yolo_loss', min_delta=0, patience=10, verbose=1)
    nan_term = tf.keras.callbacks.TerminateOnNaN()
    csv_logger = CSVLogger(log_dir + model_name + '_training_log.csv', append=True, separator=',')

    # Validation split
    val_split = 0.1
    with open(box_anno_path) as f:
        box_lines = f.readlines()
        
    np.random.seed(10101)
    np.random.shuffle(box_lines)
    np.random.seed(None)
    num_box_val = int(len(box_lines)*val_split)
    num_box_train = len(box_lines) - num_box_val
    
    print('\nYOLO Detection training on {} samples, val on {} samples'.format(num_box_train, num_box_val))
    print("--- TRAINING PARAMETERS ---")
    print("Batch size: {}\nStage1 Epochs: {}\nStage1 LR: {}\nStage2 Epochs: {}\nStage2 LR: {}\n----------\n".format(
                    batch_size, stage1_epochs, stage1_lr, stage2_epochs, stage2_lr))
    # ------------------------------------------------------------------------------------------------------------------------------
    
    # Train with frozen layers first, to get a stable loss. This step is enough to obtain a not bad model.
    model.compile(optimizer=Adam(learning_rate=stage1_lr), 
        loss={'yolo': lambda y_true, y_pred: y_pred},
        loss_weights=[1])
    model.fit(data_generator_wrapper(box_lines[:num_box_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_box_train//batch_size),
            validation_data=data_generator_wrapper(box_lines[num_box_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_box_val//batch_size),
            epochs=stage1_epochs,
            initial_epoch=start_epoch,
            callbacks=[logging, checkpoint, early_stopping, csv_logger, nan_term])
    model.save_weights(log_dir + model_name + '_trained_weights_stage1.h5')
        
    print("\nSTAGE 1 TRAINING COMPLETE\n")

# ------------------------------------
    # Unfreeze and continue training, to fine-tune. Train longer if the result is not good.
    print("Unfreezing whole model...")
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(learning_rate=stage2_lr), # recompile to apply the change
        loss={'yolo': lambda y_true, y_pred: y_pred},
        loss_weights=[1])
    model.fit(data_generator_wrapper(box_lines[:num_box_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_box_train//batch_size),
        validation_data=data_generator_wrapper(box_lines[num_box_train:], batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, num_box_val//batch_size),
        epochs=(stage1_epochs + stage2_epochs),
        initial_epoch=stage1_epochs,
        callbacks=[logging, checkpoint, early_stopping, reduce_lr, csv_logger, nan_term])
    model.save_weights(log_dir + model_name + '_weights_stage2.ckpt')
    
    print("\nSTAGE 2 TRAINING COMPLETE\n")
# -----------------------------------------

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def data_generator(box_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(box_lines)
    i = 0
    while True:
        box_image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(box_lines)
            box_image, box = get_random_yolo_data(box_lines[i], input_shape, random=True)
            box_image_data.append(box_image)
            box_data.append(box)
            i = (i+1) % n
        box_image_data = np.array(box_image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [box_image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(box_lines, batch_size, input_shape, anchors, num_classes):
    n = len(box_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(box_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    main()
