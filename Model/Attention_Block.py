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

    """
    :param input: The input of the Residual_Unit. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
    :param in_channel: The 4-th dimension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
    :param out_channel: The 4-th dimension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
    """

    # initial Attention Module parameters
    p = 1
    t = 2
    r = 1
    skip_connections = []

    # pre-activation Residual Unit
    for i in range(p):
        x = Residual_Unit(input, in_channel // 4, out_channel)

    # Trunk Branch
    for i in range(t):
        Trunck_output = Residual_Unit(x)

    # Soft Mask Branch
    ## down sampling
    x = MaxPooling2D(padding='same')(x)
    for i in range(r):
        x = Residual_Unit(x, in_channel // 4, out_channel)

    ## skip connections
    x = Residual_Unit(x, in_channel // 4, out_channel)
    skip_connections.append(x)

    ## down sampling
    x = MaxPooling2D(padding='same')(x)
    for i in range(r):
        x = Residual_Unit(x, in_channel // 4, out_channel)


    skip_connections = list(reversed(skip_connections))

    ## up sampling
    for i in range(r):
        x = Residual_Unit(x, in_channel // 4, out_channel)
    x = UpSampling2D()(x)
    
    ## skip connections
    output_soft_mask = Add()([output_soft_mask, skip_connections[i]])





