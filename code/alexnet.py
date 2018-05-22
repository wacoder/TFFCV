import tensorflow as tf

def alexnet_v2(inputs, num_classes=10, is_training=True, dropout_keep_prob=0.5):
    # code adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/alexnet.py
    regularizer = tf.keras.regularizers.l2(0.001)

    conv_1 = tf.keras.layers.Conv2D(64, (11,11), 4,input_shape=(224, 224, 3), padding='valid', activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizer, name ='conv1')(inputs)
    conv_1 = tf.keras.layers.MaxPool2D((3,3), 2,name="pool1")(conv_1)

    conv_2 = tf.keras.layers.Conv2D(192,(5, 5), padding="same",activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizer, name='conv2')(conv_1)
    conv_2 = tf.keras.layers.MaxPool2D((3,3), 2, name='pool2')(conv_2)

    conv_3 = tf.keras.layers.Conv2D(384,(3,3), padding="same",activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizer, name='conv3')(conv_2)
    conv_4 = tf.keras.layers.Conv2D(384,(3,3), padding='same',activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizer, name='conv4')(conv_3)
    conv_5 = tf.keras.layers.Conv2D(384,(3,3), padding='same',activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizer, name='conv5')(conv_4)
    
    conv_5 = tf.keras.layers.MaxPool2D((3,3),2, name='pool5')(conv_5)

    fc_6 = tf.keras.layers.Conv2D(4096, (5,5), padding='valid', activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizer, name='fc6')(conv_5)
    fc_6 = tf.keras.layers.Dropout(dropout_keep_prob, name='dropout6')(fc_6, training=is_training)

    fc_7 = tf.keras.layers.Conv2D(4096, (1,1), padding='valid', activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizer, name='fc7')(fc_6)
    fc_7 = tf.keras.layers.Dropout(dropout_keep_prob, name='dropout7')(fc_7, training=is_training)

    fc_8 = tf.keras.layers.Conv2D(num_classes, (1,1), padding='valid', activation=None, kernel_regularizer=regularizer, name='fc8')(fc_7)

    output = tf.squeeze(fc_8, axis=[1,2], name='fc_8/squeezed')
    return output










    

