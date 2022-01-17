# -*- coding:utf-8 -*-
import tensorflow as tf

from keras import backend as K
from functools import partial

class MaxPoolingWithArgmax2D(tf.keras.layers.Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, pool_size[0], pool_size[1], 1]
        padding = padding.upper()
        strides = [1, strides[0], strides[1], 1]
        output, argmax = tf.nn.max_pool_with_argmax(
                inputs,
                ksize=ksize,
                strides=strides,
                padding=padding)
        argmax = tf.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
                dim//ratio[idx]
                if dim is not None else None
                for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(tf.keras.layers.Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        #with K.tf.variable_scope(self.name):
        with tf.compat.v1.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                        input_shape[0],
                        input_shape[1]*self.size[0],

                        input_shape[2]*self.size[1],
                        input_shape[3])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                    [[input_shape[0]], [1], [1], [1]],
                    axis=0)
            batch_range = tf.keras.backend.reshape(
                    tf.range(output_shape[0], dtype='int32'),
                    shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
                mask_shape[0],
                mask_shape[1]*self.size[0],
                mask_shape[2]*self.size[1],
                mask_shape[3]
                )

def UpSample2D(pool, indx, output_shape):

    #pool_ = tf.reshape(pool, [-1])
    pool_ = pool
    #batch_range = tf.reshape(tf.range(batch_size, dtype=indx.dtype), [tf.shape(pool)[0], 1, 1, 1])
    b = tf.expand_dims(tf.ones_like(indx), -1)
    #b = tf.reshape(b, [-1, 1])
    #indx_ = tf.reshape(indx, [-1, 1])
    indx_ = tf.expand_dims(indx, -1)
    indx_ = tf.concat([b, indx_], -1)
    ret = tf.scatter_nd(indx_, pool_, shape=[tf.shape(pool)[0], output_shape[1] * output_shape[2] * output_shape[3]])
    ret = tf.reshape(ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])

    return ret

def residual_block(input):

    h = tf.keras.layers.Conv2D(8, (1,1), activation="relu", padding="same")(input)
    h = tf.keras.layers.Conv2D(8, (5,1), activation="relu", padding="same")(h)
    h = tf.keras.layers.Conv2D(8, (1,5), activation="relu", padding="same")(h)
    h = tf.keras.layers.Conv2D(16, (1,1), activation="relu", padding="same")(h)
    
    return tf.keras.layers.Add()([h, input])

def Seg_Enet(input_shape=(384, 512, 14), classes=3, batch_size=4):

    h = inputs = tf.keras.Input(input_shape, batch_size=batch_size)

    h = tf.keras.layers.Conv2D(16, (5,5), padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = residual_block(h)
    h = residual_block(h)
    h = residual_block(h)
    pool1, poo1_indx1 = MaxPoolingWithArgmax2D((2,2))(h)

    h = residual_block(pool1)
    h = residual_block(h)
    h = residual_block(h)
    pool2, poo1_indx2 = MaxPoolingWithArgmax2D((2,2))(h)

    h = residual_block(pool2)
    h = residual_block(h)
    h = residual_block(h)
    pool3, poo1_indx3 = MaxPoolingWithArgmax2D((2,2))(h)

    h = residual_block(pool3)
    h = residual_block(h)
    h = residual_block(h)
    pool4, poo1_indx4 = MaxPoolingWithArgmax2D((2,2))(h)

    h = MaxUnpooling2D((2,2))([pool4, poo1_indx4])
    h = residual_block(h)
    h = residual_block(h)
    h = residual_block(h)

    h = MaxUnpooling2D((2,2))([h, poo1_indx3])
    h = residual_block(h)
    h = residual_block(h)
    h = residual_block(h)

    h = MaxUnpooling2D((2,2))([h, poo1_indx2])
    h = residual_block(h)
    h = residual_block(h)
    h = residual_block(h)

    h = MaxUnpooling2D((2,2))([h, poo1_indx1])
    h = residual_block(h)
    h = residual_block(h)
    h = residual_block(h)

    h = tf.keras.layers.Conv2D(3, (1,1), padding="same")(h)


    return tf.keras.Model(inputs=inputs, outputs=h)

def _zero_crossings(img, kappa=.75):
    rows, cols = img.shape[:2]
    blocks = []

    for r in range(3):
        for c in range(3):
            block = img[r:rows-2+r, c:cols-2+c]
            blocks.append(block)

    min_map = blocks[0]
    max_map = blocks[0]

    for block in blocks[1:]:
        min_map = tf.math.minimum(min_map, block)
        max_map = tf.math.maximum(max_map, block)

    pos_img = img[1:rows-1, 1:cols-1] > 0.

    neg_min = tf.cast(min_map < 0., 'uint8')
    neg_min = neg_min * tf.cast(pos_img, 'uint8')

    pos_max = tf.cast(max_map > 0., 'uint8')
    pos_max = pos_max * tf.cast(tf.logical_not(pos_img), 'uint8')

    zero_cross = tf.logical_or(tf.cast(neg_min, 'bool'), tf.cast(pos_max, 'bool'))

    value_scale = 1. / tf.maximum(1., tf.reduce_max(img) - tf.reduce_min(img))
    values = value_scale * (max_map - min_map)
    values = values * tf.cast(zero_cross, 'float32')

    if kappa >= 0.:
        thresh = tf.reduce_mean(tf.abs(img)) * kappa
        threshed = tf.cast(tf.logical_not(values < thresh), 'float32')
        values = values * threshed

    return values

def laplacian(img, kappa=0.75):
    laplace_kernel = tf.constant([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=tf.float32, name='laplace_kernel')
    laplace_kernel = laplace_kernel[:, :, tf.newaxis, tf.newaxis]
    h = tf.nn.depthwise_conv2d(img[tf.newaxis, :, :, :], laplace_kernel, (1, 1, 1, 1), 'VALID')
    v = tf.map_fn(partial(_zero_crossings, kappa=kappa), h)
    return v

#import matplotlib.pyplot as plt

#img = tf.io.read_file("D:/[1]DB/[5]4th_paper_DB/crop_weed/datasets_IJRR2017/raw_aug_rgb_img/rgb_00023.png")
#img = tf.image.decode_png(img, 1)
#img = tf.image.resize(img, [224, 224]) / 255.
#img = tf.pad(img, [[2,2],[2,2],[0,0]], constant_values=0)
#img = laplacian(img)
#plt.imshow(img[0, :, :, 0], cmap="gray")
#plt.show()
