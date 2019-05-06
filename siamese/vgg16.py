import os
import sys

import numpy as np
import tensorflow as tf


"""
A VGG-16 adapted to be a branch of a siamese network. The architecture is the
same only the fully connected layers differ. The final output has a length
equal to the number of desired features. 
"""
class Vgg16:
    def __init__(self, inp, keep_prob, width_fc=512, num_features=2, 
                 load_conv_weights=False, vgg16_npy_path=None): 
        #self.image_size = image_size
        self.inp = inp 
        self.keep_prob = keep_prob
        """ inp and keep_prob are placeholders created outside """
        self.load_conv_weights = load_conv_weights
        self.width_fc = width_fc
        self.num_features = num_features
        
        if self.load_conv_weights:
            if vgg16_npy_path is None:
                path = sys.modules[self.__class__.__module__].__file__
                path = os.path.abspath(os.path.join(path, os.pardir))
                path = os.path.join(path, 'vgg16.npy')
                print(path)
                vgg16_npy_path = path
    
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            print('npy file loaded')

        #""" the placeholders """
        #self.inp = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        #self.keep_prob = tf.placeholder(tf.float32)
        
        with tf.name_scope('vgg16'):
            self.build()
            
        self.out = self.fc8


    def build(self):
        if self.load_conv_weights:
            self.center()
        else:
            """ we disregard reordering of channels and centering needed by
            vgg weights. The input is already in the range 0..1 """
            self.x = self.inp
            
            
        self.build_conv_layers()
        self.build_fc_layers()
        
        self.probs = tf.nn.softmax(self.fc8, name='probs')
        print('probs shape {}'.format(self.probs.get_shape().as_list()))


    def center(self):
        self.vgg_mean = [103.939, 116.779, 123.68]
        rgb_scaled = self.inp * 255.0
        red, green, blue = tf.split(rgb_scaled, num_or_size_splits=3, axis=3, 
                                    name='concat')
        self.x = tf.concat([blue - self.vgg_mean[0], 
                            green - self.vgg_mean[1], 
                            red - self.vgg_mean[2]], 3) # BG


    def build_conv_layers(self):
        self.conv1_1 = self._conv_layer(self.x, "conv1_1", 64)
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2", 64)
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        print('pool1 {}'.format(self.pool1.get_shape().as_list()))

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1", 128)
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2", 128)
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        print('pool2 {}'.format(self.pool2.get_shape().as_list()))

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1", 256)
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2", 256)
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3", 256)
        self.pool3 = self._max_pool(self.conv3_3, 'pool3')

        print('pool3 {}'.format(self.pool3.get_shape().as_list()))

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1", 512)
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2", 512)
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3", 512)
        self.pool4 = self._max_pool(self.conv4_3, 'pool4')

        print('pool4 {}'.format(self.pool4.get_shape().as_list()))

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1", 512)
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2", 512)
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3", 512)
        self.pool5 = self._max_pool(self.conv5_3, 'pool5')
        
        shape_pool5 = self.pool5.get_shape().as_list()
        print('pool5 {}'.format(shape_pool5))
        
        self.flat_pool5 = tf.reshape(self.pool5, [-1, np.prod(shape_pool5[1:])], 
                                     name="flat_pool5")
        print('flat_pool5 shape {}'.format(self.flat_pool5.get_shape().as_list()))


    """ The *new* fully connected layers because they are not loaded, we
    are not going to classify ImageNet and the final output size is not
    the 1024 ImageNet classes but a certain number of desired features """
    def build_fc_layers(self):        
        self.fc6 = self._fc_layer(self.flat_pool5, self.width_fc, 'fc6')
        print('fc6 shape {}'.format(self.fc6.get_shape().as_list()))
        self.relu6 = tf.nn.relu(self.fc6)
        self.dropout6 = tf.nn.dropout(self.relu6, self.keep_prob)

        self.fc7 = self._fc_layer(self.dropout6, self.width_fc, 'fc7')
        print('fc7 shape {}'.format(self.fc7.get_shape().as_list()))
        self.relu7 = tf.nn.relu(self.fc7)
        self.dropout7 = tf.nn.dropout(self.relu7, self.keep_prob)

        self.fc8 = self._fc_layer(self.dropout7, self.num_features, 'fc8')
        """ no dropout nor relu after last conv """
        print('fc8 shape {}'.format(self.fc8.get_shape().as_list()))
        

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1,2,2,1], strides=[1,2,2,1],
                              padding='SAME', name=name)


    def _get_conv_filter(self, name, depth_in, depth_out):
        if self.load_conv_weights:
            initializer = self.data_dict[name][0]
            return tf.get_variable(name+'_filter', initializer=initializer)
        else:
            #initializer = None # random uniform
            #initializer = tf.random_normal([dim_out], stddev=0.1)
            initializer = tf.contrib.layers.xavier_initializer()            
            return tf.get_variable(name+'_filter', shape=[3,3,depth_in,depth_out],
                                   initializer=initializer)


    def _get_conv_bias(self, name, depth):
        if self.load_conv_weights:
            initializer = self.data_dict[name][1]
            return tf.get_variable(name+'_biases', initializer=initializer)
        else:
            #initializer = None # random uniform
            #initializer = tf.random_normal([dim_out], stddev=0.1)
            initializer = tf.constant_initializer(0.1)            
            return tf.get_variable(name+'_biases', shape=[depth], 
                                   initializer=initializer)
        

    def _conv_layer(self, bottom, name, depth):
        with tf.variable_scope(name):
            depth_in = bottom.get_shape().as_list()[-1]
            filt = self._get_conv_filter(name, depth_in, depth)
            conv = tf.nn.conv2d(bottom, filt, [1,1,1,1], padding='SAME')
            #print 'conv', conv.get_shape().as_list()
            conv_biases = self._get_conv_bias(name, depth)
            #print 'conv_biases', conv_biases.get_shape().as_list()
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu


    def _get_fc_weights(self, bottom, dim_out, name):
        shape_bottom = bottom.get_shape().as_list()[1:]
        dim_in = np.prod(shape_bottom)
        initializer = tf.random_normal([dim_in, dim_out], stddev=0.1)
        return tf.get_variable(name+'_weights', initializer=initializer)


    def _get_fc_bias(self, dim_out, name):
        initializer = tf.random_normal([dim_out], stddev=0.1)
        return tf.get_variable(name+'_bias', initializer=initializer)


    def _fc_layer(self, bottom, dim_out, name):
        with tf.variable_scope(name):
            weights = self._get_fc_weights(bottom, dim_out, name)
            biases = self._get_fc_bias(dim_out, name)
            """ Fully connected layer. Note that the '+' operation 
            automatically broadcasts the biases """
            fc = tf.nn.bias_add(tf.matmul(bottom, weights), biases)
            return fc
            

            

if __name__ == '__main__':  
    tf.reset_default_graph()
    image_size = 64
    inp = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    keep_prob = tf.placeholder(tf.float32)
    
    with tf.variable_scope('test1'):
        vgg1 = Vgg16(inp, keep_prob, load_conv_weights=False)
        del(vgg1)

    
    with tf.variable_scope('test3') as scope:
        vgg1 = Vgg16(inp, keep_prob, load_conv_weights=False)
        scope.reuse_variables()
        vgg2 = Vgg16(inp, keep_prob, load_conv_weights=False)
        del(vgg1)
        del(vgg2)

    