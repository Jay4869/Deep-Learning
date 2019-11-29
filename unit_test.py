import unittest
import numpy as np
import tensorflow as tf
from Model import Residual_Unit
from Model import Attention_Block

class MyTestCase(unittest.TestCase):

    image = tf.random.normal(shape=[1, 224, 224, 3])

    def test_Residual_Unit(self):
        output = Residual_Unit(self.image, 64, 256)
        self.assertEqual(output.shape, [1,224,224,256])


    def test_Attention_Block(self):
        output = Attention_Block(self.image, 256, 256)
        self.assertEqual(output.shape, [1,224,224,256])

if __name__ == '__main__':
    unittest.main()
