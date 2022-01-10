# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging

import os
import tensorflow as tf
import nni
import numpy as np

LOG = logging.getLogger('mnist_keras')
TENSORBOARD_DIR = os.environ['NNI_OUTPUT_DIR']

H, W = 28, 28
NUM_CLASSES = 10

def create_mnist_model(hyper_params, input_shape=(H, W, 1), num_classes=NUM_CLASSES):
    '''
    Create simple convolutional model
    '''
    #model = tf.keras.Sequential([
    #    tf.keras.layers.Flatten(input_shape=(28, 28)),
    #    tf.keras.layers.Dense(hyper_params.get('nb_nodes'), activation='relu'),
    #    tf.keras.layers.Dense(10)
    #])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=hyper_params.get('activation_function'), input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), activation=hyper_params.get('activation_function')),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hyper_params.get('nb_nodes'), activation=hyper_params.get('activation_function')),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # FIXME: regarder l'utilisation du learning_rate
    lr = hyper_params['learning_rate']
    if hyper_params.get('optimizer') == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=hyper_params.get('metrics'))

    return model

def load_mnist_data(args):
    '''
    Load MNIST dataset
    '''
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    LOG.debug('x_train shape: %s', (x_train.shape,))
    LOG.debug('x_test shape: %s', (x_test.shape,))

    return x_train, y_train, x_test, y_test

class SendMetrics(tf.keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        LOG.debug(logs)
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])

def train(args, params):
    '''
    Train model
    '''
    x_train, y_train, x_test, y_test = load_mnist_data(args)
    model = create_mnist_model(params)

    model.fit(x_train, y_train, epochs=10)


    _, acc = model.evaluate(x_test, y_test, verbose=0)
    LOG.debug('Final result is: %d', acc)
    nni.report_final_result(acc)

def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'optimizer': 'adam',
        'learning_rate': 0.001
    }

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--batch_size", type=int, default=200, help="batch size", required=False)
    PARSER.add_argument("--epochs", type=int, default=10, help="Train epochs", required=False)
    PARSER.add_argument("--num_train", type=int, default=60000, help="Number of train samples to be used, maximum 60000", required=False)
    PARSER.add_argument("--num_test", type=int, default=10000, help="Number of test samples to be used, maximum 10000", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = generate_default_params()
        PARAMS.update(RECEIVED_PARAMS)
        # train
        train(ARGS, PARAMS)
    except Exception as e:
        LOG.exception(e)
        raise
