import unittest
import tensorflow as tf
from Module import Residual_Unit
from Module import Residual_Unit_pre
from Model import ResNet
from Module import Attention_Block
from Module import Attention_Block_NAL

class MyTestCase(unittest.TestCase):

    image = tf.random.normal(shape=[1, 224, 224, 32])

    # def test_Residual_Unit(self):
    #     output = Residual_Unit(self.image, 64, 256)
    #     self.assertEqual(output.shape, [1,224,224,256])
    #
    # def test_Residual_Unit2(self):
    #     output = Residual_Unit(self.image, 64, 256, stride=2)
    #     self.assertEqual(output.shape, [1,112,112,256])
    #
    # def test_Attention_Block(self):
    #     output = Attention_Block(self.image, skip=1)
    #     self.assertEqual(output.shape, [1,224,224,3])
    #
    # def test_Attention_Block2(self):
    #     output = Residual_Unit(self.image, 64, 256, stride=2)
    #     output = Attention_Block(output, skip=2)
    #     self.assertEqual(output.shape, [1,112,112,256])
    #
    # def test_Attention_Block_NAL(self):
    #     output = Attention_Block_NAL(self.image, skip=1)
    #     self.assertEqual(output.shape, [1,224,224,3])

    # def test_Attention_Block_NAL2(self):
    #     output = Residual_Unit(self.image, 64, 256, stride=2)
    #     output = Attention_Block_NAL(output, skip=2)
    #     self.assertEqual(output.shape, [1,112,112,256])

    def Residual_Unit_pre(self):
        output = Residual_Unit_pre(self.image, 32)
        self.assertEqual(output.shape, [1,224,224,256])

if __name__ == '__main__':
    unittest.main()
