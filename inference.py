import sys
sys.path.append('c:/etc/code/table_detection_mrcnn_tensorflow')


import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'table']

class TableConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "table"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode = 'inference', 
                             config = TableConfig(),
                             model_dir = 'log')

# Load the weights into the model.
model.load_weights(filepath = 'log/mask_rcnn_table_config_0003.h5', 
                   by_name = True)

# load the input image, convert it from BGR to RGB channel
image = cv2.imread('report_sample.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image])

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])
