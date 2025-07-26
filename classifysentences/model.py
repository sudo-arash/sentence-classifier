#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    26-Jul-2025 14:12:00

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    input = keras.Input(shape=(5751,))
    fc_1 = layers.Dense(10, name="fc_1_")(input)
    relu = layers.ReLU()(fc_1)
    fc_2 = layers.Dense(2, name="fc_2_")(relu)
    softmax = layers.Softmax()(fc_2)
    classoutput = softmax

    model = keras.Model(inputs=[input], outputs=[classoutput])
    return model
