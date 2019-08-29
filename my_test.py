
import os
import os.path as ops
import argparse
import math
import tensorflow as tf
import glog as log
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

import lanenet_merge_model
from config import global_config
from data_provider import lanenet_data_processor_test


CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--vid_path',default='data/sample.mp4', type=str, help='The path where the video is stored')

    parser.add_argument('--weights_path', default='trained/culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='False')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=1)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default='predicts/')
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()



def test_lanenet(vid_path, weights_path, use_gpu, batch_size, save_dir):

    """
    :param image_path: -> vid_path
    :param weights_path:
    :param use_gpu:
    :return:
    """
    
    test_dataset = lanenet_data_processor_test.DataSet(vid_path, batch_size)
    input_tensor = tf.placeholder(dtype=tf.int32, shape=[1, None, None, 3], name='input_tensor')
    imgs = tf.map_fn(test_dataset.process_img, input_tensor, dtype=tf.float32)
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet()
    binary_seg_ret, instance_seg_ret = net.test_inference(imgs, phase_tensor, 'lanenet_loss')
    initial_var = tf.global_variables()
    final_var = initial_var[:-1]
    saver = tf.train.Saver(final_var)
    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=sess_config)
    with sess.as_default():
        avar = 0
        sess.run(tf.global_variables_initializer())
        saver.restore(sess=sess, save_path=weights_path)
        cont = True
        while cont:
            paths = test_dataset.next_batch()
            if paths is None:
                sess.close()
                return
            

            instance_seg_image, existence_output = sess.run([binary_seg_ret, instance_seg_ret],
                                                            feed_dict={input_tensor: paths})
            # for cnt, image_name in enumerate(paths):
            #     print(image_name)
            #     parent_path = os.path.dirname(image_name)
            #     directory = os.path.join(save_dir, 'vgg_SCNN_DULR_w9', parent_path)
            #     if not os.path.exists(directory):
            #         os.makedirs(directory)
            #     file_exist = open(os.path.join(directory, os.path.basename(image_name)[:-3] + 'exist.txt'), 'w')
            print("It happened for: ", avar)
            avar+=1
            for cnt_img in range(4):
                cv2.imwrite('predicts/'+os.path.join(str(cnt_img + 1) + '_avg.png'),(instance_seg_image[0, :, :, cnt_img + 1] * 255).astype(int))
        #     if existence_output[cnt, cnt_img] > 0.5:
        #         file_exist.write('1 ')
        #     else:
        #         file_exist.write('0 ')
        # file_exist.close()
    sess.close()
    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    save_dir = os.path.join('predicts')
    if args.save_dir is not None:
        save_dir = args.save_dir

    # img_name = []
    # with open(str(args.image_path), 'r') as g:
    #     for line in g.readlines():
    #         img_name.append(line.strip())

    test_lanenet(args.vid_path, args.weights_path, args.use_gpu, args.batch_size, save_dir)