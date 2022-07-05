import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import gphoto2 as gp
import time
import os


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
# flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
# flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

splitline = 10 * '='


def capture_image(num):
    camera = gp.Camera()
    camera.init()
    print(splitline + 'Capturing image' + splitline)
    time_start = time.time()
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
    print('use {} seconds for capturing'.format(time.time() - time_start))
    print(splitline + 'Camera file path: {0}/{1}'.format(file_path.folder, file_path.name) + splitline)
    target = os.path.join('/home/jing/WeedAI_Linux/YOLOv4-TRT/tensorflow-yolov4-tflite/fotos',
                          'Pic-' + str(num) + '.jpg')
    print(splitline + 'Copying image to', target + splitline)
    camera_file = camera.file_get(
        file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    camera_file.save(target)

    camera.exit()
    return target


def main(_argv):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    # image_path = FLAGS.image

    # load tf-trt model
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # capture image and detect
    # logging.basicConfig(
    #     format='%(levelname)s: %(name)s: %(message)s', level=logging.WARNING)
    # callback_obj = gp.check_result(gp.use_python_logging())
    print('Press Enter to continue to take picture. Enter EXIT to kill program.')

    num = 1
    while True:
        input_str = input()
        if input_str == '':

            image_path = capture_image(num)
            time_cap_end = time.time()

            # detecting
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
            image_data = cv2.resize(original_image, (input_size, input_size))
            image_data = image_data / 255.
            # image_data = image_data[np.newaxis, ...].astype(np.float32)

            images_data = []
            for i in range(1):
                images_data.append(image_data)
            images_data = np.asarray(images_data).astype(np.float32)

            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            print('inference use {} seconds'.format(time.time() - time_cap_end))
            image = utils.draw_bbox(original_image, pred_bbox)
            # image = utils.draw_bbox(image_data*255, pred_bbox)
            image = Image.fromarray(image.astype(np.uint8))
            image.show()
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            cv2.imwrite('/home/jing/WeedAI_Linux/YOLOv4-TRT/tensorflow-yolov4-tflite/fotos/results/Result-' + str(num) + '.jpg', image)
            print('Inference finish! Press Enter to continue to take picture. Enter EXIT to kill program.')
            num = num + 1

        elif input_str == 'EXIT':
            print('kill the program')
            break
        else:
            print('Press Enter to continue to take picture. Enter EXIT to kill program.')
    return 0


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
