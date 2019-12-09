import unittest
import numpy as np
import tensorflow as tf
from Model import Residual_Unit
from Model import Attention_Block
from Model import AttentionResNet56

class MyTestCase(unittest.TestCase):

    image = tf.random.normal(shape=[1, 224, 224, 3])

    def test_Residual_Unit(self):
        output = Residual_Unit(self.image, 64, 256)
        self.assertEqual(output.shape, [1,224,224,256])

    def test_Residual_Unit2(self):
        output = Residual_Unit(self.image, 64, 256, stride=2)
        self.assertEqual(output.shape, [1,112,112,256])

    def test_Attention_Block(self):
        image = tf.random.normal(shape=[1,14,14,256])
        output = Attention_Block(image, skip=1)
        self.assertEqual(output.shape, [1,14,14,256])

    def test_Attention_Block2(self):
        output = Residual_Unit(self.image, 64, 256, stride=2)
        output = Attention_Block(output, skip=2)
        self.assertEqual(output.shape, [1,112,112,256])


if __name__ == '__main__':
    unittest.main()
