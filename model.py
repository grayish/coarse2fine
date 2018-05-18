import tensorflow as tf

import module
import module.layers as layers


class Model:
    def __init__(self, features, joints, z_resolutions):
        self.features = features
        self.joints = joints
        self.z_resolutions = z_resolutions

        with tf.variable_scope('input'):
            self.images = tf.placeholder(
                name='image',
                dtype=tf.float32,
                shape=[None, 256, 256, 3])
            self.voxel_groundtruth = [
                tf.placeholder(
                    name='voxel_groundtruth_%d' % stage,
                    dtype=tf.float32,
                    shape=[None, 64, 64, self.joints * z_resolution])
                for stage, z_resolution in enumerate(self.z_resolutions)
            ]
            self.is_training = tf.placeholder(
                name='train',
                dtype=tf.bool,
                shape=())
            
        with tf.variable_scope('compress'):
            with tf.variable_scope('conv_bn_relu'):
                net = layers.conv(inputs=self.images, ksize=7, kchannel=64, kstride=2)  # 128 * 128 * 64
                net = layers.bn(inputs=net, is_training=self.is_training)
                net = layers.relu(inputs=net)

            net = module.bottleneck(inputs=net, kchannel=128, is_training=self.is_training, name='A')  # 128 * 128 * 128
            net = layers.pool(inputs=net)  # 64 * 64 * 128
            net = module.bottleneck(inputs=net, kchannel=128, is_training=self.is_training, name='B')  # 64 * 64 * 128
            net = module.bottleneck(inputs=net, kchannel=self.features, is_training=self.is_training, name='C')  # 64 * 64 * 256

        self.voxels = []
        for stage, z_resolution in enumerate(self.z_resolutions):
            with tf.variable_scope('hourglass_' + str(stage)):
                prev = tf.identity(net)
                net = module.hourglass(inputs=net, is_training=self.is_training)  # 64 * 64 * 256

                with tf.variable_scope('inter_hourglasses'):
                    net = module.bottleneck(inputs=net, is_training=self.is_training)  # 64 * 64 * 256
                    net = layers.conv(inputs=net, ksize=1, kchannel=self.features)  # 64 * 64 * 256
                    net = layers.bn(inputs=net, is_training=self.is_training)
                    net = layers.relu(inputs=net)

                with tf.variable_scope('voxel'):
                    voxel = layers.conv(inputs=net, ksize=1, kchannel=self.joints * z_resolution)  # 64 * 64 * joint*z_resolution
                    self.voxels.append(voxel)

                net = layers.conv(inputs=net, ksize=1, kchannel=self.features, name='inter') \
                      + layers.conv(inputs=voxel, ksize=1, kchannel=self.features, name='voxel') \
                      + prev  # 64 * 64 * 256
            for z in range(z_resolution):
                tf.summary.image('voxel_%dD_%02d' % (z_resolution, z+1), self.voxels[stage][:, :, z])
        tf.summary.image('RGB', self.images)