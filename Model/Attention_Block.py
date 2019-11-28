from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Multiply
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
        Trunck_output = Residual_Unit(x, in_channel // 4, out_channel)

    # Soft Mask Branch
    ## two times down sampling
    x = MaxPooling2D(padding='same')(x)
    for i in range(r):
        x = Residual_Unit(x, in_channel // 4, out_channel)

    ## skip connections
    skip_connections = Residual_Unit(x, in_channel // 4, out_channel)

    x = MaxPooling2D(padding='same')(x)
    for i in range(r):
        x = Residual_Unit(x, in_channel // 4, out_channel)

    ## two times up sampling
    for i in range(r):
        x = Residual_Unit(x, in_channel // 4, out_channel)
    x = UpSampling2D()(x)

    ## skip connections
    x = Add()([x, skip_connections])

    for i in range(r):
        x = Residual_Unit(x, in_channel // 4, out_channel)
    x = UpSampling2D()(x)

    ## output
    x = Conv2D(in_channel, (1, 1))(x)
    x = Conv2D(in_channel, (1, 1))(x)
    x = Activation('sigmoid')(x)

    # Attention: (1 + soft_mask_output) * Trunck_output
    soft_mask_output = Lambda(lambda x: x + 1)(x)
    output = Multiply()([soft_mask_output, Trunck_output])  #

    # Last Residual Block
    for i in range(p):
        output = Residual_Unit(output, in_channel // 4, out_channel)

    return output







