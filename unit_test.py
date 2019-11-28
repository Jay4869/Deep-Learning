import unittest
import numpy as np
import tensorflow as tf
from Model import Residual_Unit

class MyTestCase(unittest.TestCase):
    def test_Residual_Unit(self):
        image = tf.random.normal(shape=[1, 224, 224, 3])
        output = Residual_Unit(image, 64, 256)
        self.assertEqual(output.shape, [1,224,224,256])


if __name__ == '__main__':
    unittest.main()
