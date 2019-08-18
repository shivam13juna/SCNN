
import tensorflow as tf
import numpy as np
import cv2

from config import global_config

CFG = global_config.cfg
VGG_MEAN = [123.68, 116.779, 103.939]


class DataSet(object):


    def __init__(self, vid_path, batch_size):
        """
        dataset_info_file -> image ->vid_path
        _img_list -> image

        :param dataset_info_file:
        """
        # self._dataset_info_file = dataset_info_file
        self.vidObj = cv2.VideoCapture(vid_path) 
  

        self.success, self.image = self.vidObj.read() 
        self._batch_size = batch_size
        self._next_batch_loop_count = 0

    def __len__(self):
        return self._len

    @staticmethod
    def process_img(img_decoded):
        # img_decoded = tf.image.decode_jpeg(img_path, channels=3)
        # img_decoded = tf.squeeze(img_decoded)
        # img_decoded = tf.cast(img_decoded, tf.string)
        # img_decoded = tf.image.decode_jpeg(img_decoded, channels=3)

        img_resized = tf.image.resize_images(img_decoded, [CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH])
        img_casted = tf.cast(img_resized, tf.float32)
        img_casted = tf.subtract(img_casted, VGG_MEAN) 
        return  img_casted
# 
    def next_batch(self):
        """
        :return:
        """
        self.success, self.image = self.vidObj.read() 
        self.image = np.expand_dims(self.image, 0)
        if self.success:
            return self.image
        else:
            return 0
