import os
import random
import shutil
import tarfile
import cv2
import numpy as np
#from keras.utils import Sequence
from tensorflow.python.keras.utils.data_utils import Sequence
#import keras



class DataGen(Sequence):
    def __init__(self, id_list, landmark_to_idx, batch_size=128, verbose=1):
        self.batch_size=batch_size
        self.id_list = id_list
        self.landmark_to_idx = landmark_to_idx


    def __getitem__(self, index):
        batch_id_list = random.sample(self.id_list, self.batch_size)
        landmark_to_idx = self.landmark_to_idx
        #num_classes = self.num_classes
        
        output = []
        label_idx = []
        for ix, ids in enumerate(batch_id_list):
            img_id = ids[0]
            ldmk_id = ids[1]
            path = 'train/'+str(ldmk_id)+'/'+img_id+'.jpg'
            try: 
                im = cv2.imread(path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                if im.size != 0:
                    output.append(im)
                    ldmk_idx = landmark_to_idx[ldmk_id]
                    label_idx.append(ldmk_idx)
            except:
                continue
        
        x = np.array(output)
        y = np.zeros((len(output), NUM_CLASSES))
        for i in range(len(label_idx)):
            y[i,label_idx[i]] = 1.
        
        return x,y
            
    def on_epoch_end(self):
        return

    def __len__(self):
        #return len(valid_urls_list) // self.batch_size
        return 10