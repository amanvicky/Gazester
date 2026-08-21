import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import transformation

# Use TF1 compat mode
tf1 = tf.compat.v1
tf1.disable_eager_execution()

img_crop = 3


def batch_norm(x, train_phase, name='bn_layer'):
    with tf1.variable_scope(name):
        channels = x.shape[-1]
        beta = tf1.get_variable('beta', [channels],
                                initializer=tf.zeros_initializer())
        gamma = tf1.get_variable('gamma', [channels],
                                 initializer=tf.ones_initializer())
        moving_mean = tf1.get_variable('moving_mean', [channels],
                                       initializer=tf.zeros_initializer(),
                                       trainable=False)
        moving_var = tf1.get_variable('moving_variance', [channels],
                                      initializer=tf.ones_initializer(),
                                      trainable=False)

        def train_fn():
            mean, var = tf.nn.moments(x, axes=[0, 1, 2])
            update_mean = tf1.assign(moving_mean,
                                    moving_mean * 0.9 + mean * 0.1)
            update_var = tf1.assign(moving_var,
                                   moving_var * 0.9 + var * 0.1)
            with tf.control_dependencies([update_mean, update_var]):
                return tf.nn.batch_normalization(x, mean, var, beta, gamma,
                                                1e-5)

        def eval_fn():
            return tf.nn.batch_normalization(x, moving_mean, moving_var,
                                           beta, gamma, 1e-5)

        return tf.cond(train_phase, train_fn, eval_fn)


def conv2d(inputs, filters, kernel_size, name='conv2d'):
    with tf1.variable_scope(name):
        in_channels = inputs.shape[-1]
        shape = [kernel_size[0], kernel_size[1], in_channels, filters]
        weights = tf1.get_variable('weights', shape,
                                   initializer=tf1.glorot_uniform_initializer())
        biases = tf1.get_variable('biases', [filters],
                                  initializer=tf.zeros_initializer())
        out = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1],
                          padding='SAME')
        return tf.nn.bias_add(out, biases)


def dense(inputs, units, name='dense'):
    with tf1.variable_scope(name):
        in_features = inputs.shape[-1]
        weights = tf1.get_variable('weights', [in_features, units],
                                   initializer=tf1.glorot_uniform_initializer())
        biases = tf1.get_variable('biases', [units],
                                  initializer=tf.zeros_initializer())
        return tf.matmul(inputs, weights) + biases


def cnn_blk(inputs, filters, kernel_size, phase_train, name='cnn_blk'):
    with tf1.variable_scope(name):
        cnn = conv2d(inputs, filters, kernel_size, name='cnn')
        act = tf.nn.relu(cnn, name='act')
        ret = batch_norm(act, phase_train)
        return ret


def dnn_blk(inputs, nodes, name='dnn_blk'):
    with tf1.variable_scope(name):
        dnn = dense(inputs, nodes, name='dnn')
        ret = tf.nn.relu(dnn, name='act')
        return ret


def gen_agl_map(inputs, height, width, feature_dims):
    with tf.name_scope('gen_agl_map'):
        batch_size = tf.shape(inputs)[0]
        ret = tf.reshape(
            tf.tile(inputs, tf.constant([1, height * width])),
            [batch_size, height, width, feature_dims]
        )
        return ret


def encoder(inputs, height, width, tar_dim):
    with tf1.variable_scope('encoder'):
        dnn_blk_0 = dnn_blk(inputs, 16, name='dnn_blk_0')
        dnn_blk_1 = dnn_blk(dnn_blk_0, 16, name='dnn_blk_1')
        dnn_blk_2 = dnn_blk(dnn_blk_1, tar_dim, name='dnn_blk_2')
        agl_map = gen_agl_map(dnn_blk_2, height, width, tar_dim)
        return agl_map


def apply_lcm(batch_img, light_weight):
    with tf.name_scope('apply_lcm'):
        img_wgts, pal_wgts = tf.split(light_weight, [1, 1], 3)
        img_wgts = tf.tile(img_wgts, [1, 1, 1, 3])
        pal_wgts = tf.tile(pal_wgts, [1, 1, 1, 3])
        palette = tf.ones(tf.shape(batch_img), dtype=tf.float32)
        ret = tf.add(tf.multiply(batch_img, img_wgts),
                    tf.multiply(palette, pal_wgts))
        return ret


def trans_module(inputs, structures, phase_train, name='trans_module'):
    with tf1.variable_scope(name):
        cnn_blk_0 = cnn_blk(inputs, structures['depth'][0],
                           structures['filter_size'][0],
                           phase_train, name='cnn_blk_0')
        cnn_blk_1 = cnn_blk(cnn_blk_0, structures['depth'][1],
                           structures['filter_size'][1],
                           phase_train, name='cnn_blk_1')
        cnn_blk_2 = cnn_blk(tf.concat([cnn_blk_0, cnn_blk_1], axis=3),
                           structures['depth'][2],
                           structures['filter_size'][2],
                           phase_train, name='cnn_blk_2')
        cnn_blk_3 = cnn_blk(tf.concat([cnn_blk_0, cnn_blk_1, cnn_blk_2], axis=3),
                           structures['depth'][3],
                           structures['filter_size'][3],
                           phase_train, name='cnn_blk_3')
        cnn_4 = conv2d(cnn_blk_3, structures['depth'][4],
                      structures['filter_size'][4], name='cnn_4')
        return cnn_4


def lcm_module(inputs, structures, phase_train, name='lcm_module'):
    with tf1.variable_scope(name):
        cnn_blk_0 = cnn_blk(inputs, structures['depth'][0],
                           structures['filter_size'][0],
                           phase_train, name='cnn_blk_0')
        cnn_blk_1 = cnn_blk(cnn_blk_0, structures['depth'][1],
                           structures['filter_size'][1],
                           phase_train, name='cnn_blk_1')
        cnn_2 = conv2d(cnn_blk_1, structures['depth'][2],
                      structures['filter_size'][2], name='cnn_2')
        lcm_map = tf.nn.softmax(cnn_2)
        return lcm_map


def inference(input_img, input_fp, input_agl, phase_train, conf):
    """Build the FLX-Gaze model.

    Args:
        input_img: Input eye image [batch, height, width, 3]
        input_fp: Anchor/fingerprint maps [batch, height, width, 12]
        input_agl: Target gaze angle [batch, 2]
        phase_train: Boolean for batch normalization
        conf: Configuration object

    Returns:
        img_pred: Predicted gaze-corrected image
        flow_raw: Raw flow field
        lcm_map: Light correction map
    """
    corse_layer = {
        'depth': (32, 64, 64, 32, 16),
        'filter_size': ([5, 5], [3, 3], [3, 3], [3, 3], [1, 1])
    }
    fine_layer = {
        'depth': (32, 64, 32, 16, 4),
        'filter_size': ([5, 5], [3, 3], [3, 3], [3, 3], [1, 1])
    }
    lcm_layer = {
        'depth': (8, 8, 2),
        'filter_size': ([3, 3], [3, 3], [1, 1])
    }

    with tf1.variable_scope('warping_model'):
        agl_map = encoder(input_agl, conf.height, conf.width,
                         conf.encoded_agl_dim)
        igt_inputs = tf.concat([input_img, input_fp, agl_map], axis=3)

        with tf1.variable_scope('warping_module'):
            # Coarse module
            resized_igt_inputs = tf.nn.avg_pool2d(
                igt_inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME'
            )
            coarse_raw = trans_module(resized_igt_inputs, corse_layer,
                                     phase_train, name='coarse_level')
            coarse_act = tf.nn.tanh(coarse_raw)
            coarse_resize = tf.image.resize(
                coarse_act, (conf.height, conf.width),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            coarse_out = tf.nn.avg_pool2d(
                coarse_resize, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                padding='SAME'
            )

            # Fine module
            fine_input = tf.concat([igt_inputs, coarse_out], axis=3)
            fine_out = trans_module(fine_input, fine_layer, phase_train,
                                   name='fine_level')
            flow_raw, lcm_input = tf.split(fine_out, [2, 2], 3)

        flow = tf.nn.tanh(flow_raw)
        cfw_img = transformation.apply_transformation(
            flows=flow, img=input_img, num_channels=3
        )

        # LCM module
        lcm_map = lcm_module(lcm_input, lcm_layer, phase_train,
                            name='lcm_module')
        img_pred = apply_lcm(batch_img=cfw_img, light_weight=lcm_map)

        return img_pred, flow_raw, lcm_map


def dist_loss(y_pred, y_, method="L2"):
    with tf1.variable_scope('img_dist_loss'):
        if method == "L2":
            loss = tf.sqrt(tf.clip_by_value(
                tf.reduce_sum(tf.square(y_pred - y_), axis=3, keepdims=True),
                1e-10, 1000
            ))
        elif method == "MAE":
            loss = tf.abs(y_pred - y_)

        loss = loss[:, img_crop:(-1) * img_crop, img_crop:(-1) * img_crop, :]
        loss = tf.reduce_sum(loss, axis=[1, 2, 3])
        return tf.reduce_mean(loss, axis=0)


def TVloss(inputs):
    with tf1.variable_scope('TVloss'):
        dinputs_dx = inputs[:, :-1, :, :] - inputs[:, 1:, :, :]
        dinputs_dy = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
        dinputs_dx = tf.pad(dinputs_dx, [[0, 0], [0, 1], [0, 0], [0, 0]],
                           "CONSTANT")
        dinputs_dy = tf.pad(dinputs_dy, [[0, 0], [0, 0], [0, 1], [0, 0]],
                           "CONSTANT")
        tot_var = tf.add(tf.abs(dinputs_dx), tf.abs(dinputs_dy))
        tot_var = tf.reduce_sum(tot_var, axis=3, keepdims=True)
        return tot_var


def center_weight(shape, base=0.005, boundary_penalty=3.0):
    with tf1.variable_scope('center_weight'):
        temp = boundary_penalty - base
        x = tf.pow(tf.abs(tf.linspace(-1.0, 1.0, shape[1])), 8)
        y = tf.pow(tf.abs(tf.linspace(-1.0, 1.0, shape[2])), 8)
        X, Y = tf.meshgrid(y, x)
        X = tf.expand_dims(X, axis=2)
        Y = tf.expand_dims(Y, axis=2)
        dist2cent = temp * tf.sqrt(
            tf.reduce_sum(tf.square(tf.concat([X, Y], axis=2)), axis=2)
        ) + base
        dist2cent = tf.expand_dims(
            tf.tile(tf.expand_dims(dist2cent, axis=0), [shape[0], 1, 1]),
            axis=3
        )
        return dist2cent


def TVlosses(eye_mask, ori_img, flow, lcm_map):
    with tf1.variable_scope('TVlosses'):
        TV_flow = TVloss(flow)
        img_gray = tf.reduce_mean(ori_img, axis=3, keepdims=True)
        ones = tf.ones(shape=tf.shape(img_gray))
        bright = ones - img_gray
        eye_mask = tf.expand_dims(eye_mask, axis=3)
        weights = tf.multiply(bright, eye_mask)
        TV_eye = tf.multiply(weights, TV_flow)

        lid_mask = ones - eye_mask
        TV_lid = tf.multiply(lid_mask, TV_flow)

        TV_eye = tf.reduce_sum(TV_eye, axis=[1, 2, 3])
        TV_lid = tf.reduce_sum(TV_lid, axis=[1, 2, 3])

        dist2cent = center_weight(tf.shape(lcm_map))
        TV_lcm = dist2cent * TVloss(lcm_map)
        TV_lcm = tf.reduce_sum(TV_lcm, axis=[1, 2, 3])

        return (tf.reduce_mean(TV_eye, axis=0),
                tf.reduce_mean(TV_lid, axis=0),
                tf.reduce_mean(TV_lcm, axis=0))


def lcm_adj(lcm_wgt):
    dist2cent = center_weight(tf.shape(lcm_wgt))
    with tf1.variable_scope('lcm_adj'):
        _, loss = tf.split(lcm_wgt, [1, 1], 3)
        loss = tf.reduce_sum(tf.abs(loss) * dist2cent, axis=[1, 2, 3])
        return tf.reduce_mean(loss, axis=0)


def loss(img_pred, img_, eye_mask, input_img, flow, lstm_wgt,
         loss_combination='l2sc'):
    with tf1.variable_scope('losses'):
        loss_img = dist_loss(img_pred, img_)
        loss_eyeball, loss_eyelid, loss_lcm = TVlosses(
            eye_mask, input_img, flow, lstm_wgt
        )
        loss_lcm_adj = lcm_adj(lstm_wgt)

        if loss_combination == 'l2sc':
            total = (loss_img + loss_eyeball + loss_eyelid +
                    loss_lcm_adj + loss_lcm)
        elif loss_combination == 'l2s':
            total = loss_img + loss_eyeball + loss_eyelid
        else:
            total = loss_img

        tf.add_to_collection('losses', total)
        return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss_img
