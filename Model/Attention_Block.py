from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Lambda
from .Residual_Unit import Residual_Unit

"""
Attention_Block
input >> Max Pooling >> Residual Block (p)
      >> Soft Mask Branch * Trunk Branch + identity
      >> Residual Block (p)
      >> output
"""

def Attention_Block(input, in_channel, out_channel, encoder_depth=1):

    # initial Attention Module parameters
    p = 1
    t = 2
    r = 1

    x = MaxPooling2D(input, 3, padding='valid', strides=2)

    for i in range(r):
        x = Residual_Unit(x)

    # Soft Mask Branch
    x = MaxPooling2D(input, 3, padding='valid', strides=2)

    # Trunk Branch
    for i in range(t):
        x = Residual_Unit(x)

