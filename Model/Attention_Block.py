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
      >> (1 + Soft Mask Branch) * Trunk Branch
      >> Residual Block (p)
      >> output
      
Soft Mask Branch
input >> down sampling >> skip connections >> down sampling
      >> up sampling >> skip connections >> up sampling
      >> add skip connections
      >> conv(1,1) >> conv(1,1) >> sigmoid
"""

def Attention_Block(input, skip):

    """
    :param input: The input of the Residual_Unit. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
    :param in_channel: The 4-th dimension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
    """

    # initial Attention Module parameters
    p = 1
    t = 2
    r = 1
    skip_connections = []
    # calculate input and output channel based on previous layers
    out_channel = input.shape[-1]
    in_channel = out_channel // 4

    # pre-activation Residual Unit
    for _ in range(p):
        x = Residual_Unit(input, in_channel, out_channel)

    # Trunk Branch
    for _ in range(t):
        Trunck_output = Residual_Unit(x, in_channel, out_channel)

    # Soft Mask Branch
    ## 1st down sampling
    x = MaxPooling2D(padding='same')(x)
    for _ in range(r):
        x = Residual_Unit(x, in_channel, out_channel)

    if x.shape[1] % 4 == 0:
        for i in range(skip-1):
            ## skip connections
            skip_connections.append(Residual_Unit(x, in_channel, out_channel))

            ## 2rd down sampling
            x = MaxPooling2D(padding='same')(x)
            for _ in range(r):
                x = Residual_Unit(x, in_channel, out_channel)

        skip_connections = list(reversed(skip_connections))

        for i in range(skip-1):
            ## 1st up sampling
            for _ in range(r):
                x = Residual_Unit(x, in_channel, out_channel)
            x = UpSampling2D()(x)

            # skip connections
            x = Add()([x, skip_connections[i]])

    ## 2rd up samplping
    for i in range(r):
        x = Residual_Unit(x, in_channel, out_channel)
    x = UpSampling2D()(x)

    ## output
    x = Conv2D(out_channel, (1, 1))(x)
    x = Conv2D(out_channel, (1, 1))(x)
    soft_mask_output = Activation('sigmoid')(x)

    # Attention: (1 + soft_mask_output) * Trunck_output
    soft_mask_output = Lambda(lambda x: x + 1)(soft_mask_output)
    output = Multiply()([soft_mask_output, Trunck_output])

    # Last Residual Block
    for i in range(p):
        output = Residual_Unit(output, in_channel, out_channel)

    return output







