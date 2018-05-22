import tensorflow as tf
import numpy as np
import cv2
import os


def read_and_decode(example):
    features = tf.parse_single_example(
        example,
        # Defaults are not specified since both keys are required.
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    # image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    # image = tf.image.resize_images(image, [64, 64])
    # image = tf.cast(image, tf.uint8)
    # image.set_shape([mnist.IMAGE_PIXELS])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    # image = image * (1. / 255) - 0.5

    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([32 * 32 * 3])
    image = tf.reshape(image, [32, 32, 3])
    # image = tf.reverse(image, axis=[-1])
    #image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, [224,224])
    image = tf.image.per_image_standardization(image)
    # image = tf.cast(image,tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = features['label']
    return image, label


def inputs(path, batch_size, num_epochs, allow_smaller_final_batch=False):
    """Reads input data num_epochs times.
    Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None
    # filename = os.path.join(FLAGS.train_dir,
    #                       TRAIN_FILE if train else VALIDATION_FILE)

    with tf.name_scope('input'):
        files = tf.data.Dataset.list_files(path)
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.map(map_func=read_and_decode, num_parallel_calls=4)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(1000)

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        return next_element


def get_files_name(path):
    list = os.listdir(path)
    result = []
    for line in list:
        file_path = os.path.join(path, line)
        if os.path.isfile(file_path):
            result.append(file_path)
    return result

# define the image augumentation 
def image_transform(img, ang_range, shear_range, trans_range):
    # rotation
    ang_rot = np.random.uniform(ang_range) - ang_range/2
    rows, cols, chs = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2, rows/2), ang_rot, 1)
    
    # translation
    trans_x = trans_range*np.random.uniform() - trans_range/2
    trans_y = trans_range*np.random.uniform() - trans_range/2
    Trans_M = np.float32([[1,0,trans_x],[0,1,trans_y]])

    # shear
    tr_src = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_range*np.random.uniform() - shear_range/2
    pt2 = 20+shear_range*np.random.uniform() - shear_range/2
    tr_dst = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    Shear_M = cv2.getAffineTransform(tr_src, tr_dst)


    # flip on horizontal 
    if np.random.uniform() < 0.2:
        img = cv2.flip(img, 1)
    # Affine transform
    img = cv2.warpAffine(img, Rot_M, (cols, rows), borderValue=(255,255,255))
    img = cv2.warpAffine(img, Trans_M, (cols, rows), borderValue=(255,255,255))
    img = cv2.warpAffine(img, Shear_M, (cols, rows), borderValue=(255,255,255))

    image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform(0.7, 0.9)
    image_hsv[:,:,2] = random_bright*image_hsv[:,:,2]

    img = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    return img