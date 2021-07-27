import sys
sys.path.append('c:/etc/code/table_detection_mrcnn_tensorflow')

import numpy as np
import os
import json
import skimage
import tensorflow as tf

import mrcnn
import mrcnn.utils
import mrcnn.config
from mrcnn.model import MaskRCNN


        
class TableConfig(mrcnn.config.Config):
    
    NAME = 'table_config'    
    LEARNING_RATE = 1e-4
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    
    
    


class TablebankDataset(mrcnn.utils.Dataset):
    
    def _load_data(self, file_name):
        with open(file_name, 'r') as f:
            file = json.loads(f.read())
            
        image_info, annotations = file['images'], file['annotations']
        info = []
        for image, annotation in zip(image_info, annotations):
            info.append(
                {
                    'image_id': image['id'],
                    'file_name': image['file_name'],
                    'annotation':
                        {
                            'width': image['width'],
                            'height': image['height'],
                            'bbox': annotation['bbox']
                        }
                 }
                )
            
        return info
            

    def load_dataset(self, file_path, is_train = True):
        self.add_class('dataset', 1, 'table')
        self.images_directory = file_path + 'images/'
        self.annotations_directory = file_path + 'annotations/'
        
        if is_train:
            info_dictionary = self._load_data(self.annotations_directory + 'tablebank_latex_train.json')
        else:
            info_dictionary = self._load_data(self.annotations_directory + 'tablebank_latex_val.json')
        
        for info in info_dictionary:
            image_path = self.images_directory + info['file_name']
            self.add_image('dataset', image_id = info['image_id'], path = image_path, annotation = info['annotation'])
            
    
        
    def load_mask(self, image_id):
        info = self.image_info[(image_id - 1)]
        boxes, w, h = info['annotation']['bbox'], info['annotation']['width'], info['annotation']['height']
        
        masks = np.zeros([h, w, 1], dtype = 'uint8')
        
        class_ids = list()
        # for i in range(len(boxes)):
        #     box = boxes[i]
        #     row_s, row_e = box[1], box[3]
        #     col_s, col_e = box[0], box[2]
        #     masks[row_s:row_e, col_s:col_e, i] = 1
        #     class_ids.append(self.class_names.index('table'))
        
        row_s, row_e = boxes[1], boxes[3]
        col_s, col_e = boxes[0], boxes[2]
        masks[row_s:row_e, col_s:col_e, 0] = 1
        class_ids.append(self.class_names.index('table'))
        
        return masks, np.asarray(class_ids, dtype = 'int32')
    
            

    
if __name__ == '__main__':
    
    PATH = 'c:/etc/code/table_detection_mrcnn_tensorflow/dataset/table_image/TableBank/Detection/'
    
    train_dataset = TablebankDataset()
    train_dataset.load_dataset(PATH)
    train_dataset.prepare()
    
    val_dataset = TablebankDataset()
    val_dataset.load_dataset(PATH, is_train = False)
    val_dataset.prepare()
    
    config = TableConfig()
    
    model = mrcnn.model.MaskRCNN(mode = 'training',
                                 config = config,
                                 model_dir = r'MASK-RCNN-TF2/log'
                                 )
    
    model.train(train_dataset, val_dataset, learning_rate = config.LEARNING_RATE, epochs = 5, layers = 'heads')
    model.keras_model.summary()
            
        