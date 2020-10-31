# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description: 进行更细粒度的测试，将深度学习模型进行分布式并行处理，最终得出结果。
'''
from Executer import utils
class operation:

    def __init__(self, operation_id, generation_operation_model, input_shape, weights_dict):
        import numpy as np

        self.operation_id = operation_id
        self.operation_model = generation_operation_model(input_shape)
        self.input_shape = input_shape

        'load the weight'
        weights = utils.weights
        for name in weights:
            try:
                if self.operation_model.get_layer(name) != None:
                    self.operation_model.get_layer(name).set_weights(weights_dict[name])
            except Exception as e:
                pass
        if type(input_shape) == list:
            testdata = []
            for tmp in input_shape:
                testdata.append([np.zeros(shape=tmp, dtype=np.float32)])
            self.operation_model.predict(testdata)
            pass
        else:
            self.operation_model.predict(np.array([np.zeros(shape=input_shape, dtype=np.float32)]))

    def excute(self, input):
        import numpy as np

        x_input = input
        # if np.shape(input)[0] == self.input_shape[0]:
        #     x_input = [input]
        if type(self.input_shape) != list:
            x_input = np.array(x_input)

        if type(self.input_shape) == list:
            input_data = []
            # print("the raw shape of the input is {0} of operation {1}".format(np.shape(input),
            #                                                                   self.operation_id))
            for i in range(len(self.input_shape)):
                # print("operation {0} the input shape is {1}".format(self.operation_id,
                #                                                     np.shape(input[i])))
                # x_input.append(np.array(input[i]))
                input_data.append(input[i])
                # print("operation {0} the input shape is {1}".format(self.operation_id,
                #                                                     np.shape(input_data[i])))

            # print("the operation {0} input shape is:{1}".format(self.operation_id, np.shape(x_input)))
            embedding = self.operation_model.predict(input_data)
            return embedding
        # print("the operation {0} input shape is:{1}".format(self.operation_id, np.shape(x_input)))
        embedding = self.operation_model.predict(x_input)
        return embedding

class excuteDistributedDeepLearningAgent:


    def __func0__(self, input_shape):
        from keras.layers import Conv2D, ZeroPadding2D, Activation, Input
        from keras.models import Model
        from keras.layers.normalization import BatchNormalization
        from keras.layers.pooling import MaxPooling2D
        from keras.layers.core import Lambda
        from Executer.utils import LRN2D

        if input_shape == None:
            input_shape = (96, 96, 3)

        myInput = Input(shape=input_shape)
        x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)
        x = Lambda(LRN2D, name='lrn_1')(x)
        x = Conv2D(64, (1, 1), name='conv2')(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(192, (3, 3), name='conv3')(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
        x = Activation('relu')(x)
        x = Lambda(LRN2D, name='lrn_2')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)

        model = Model(inputs=myInput, outputs=x)
        return model

    def __func1__(self, input_shape):
        from keras.layers import Conv2D, ZeroPadding2D, Activation, Input
        from keras.models import Model
        from keras.layers.normalization import BatchNormalization

        x = Input(shape=input_shape)
        inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
        inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
        inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
        inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
        inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

        model = Model(inputs=x, outputs=inception_3a_3x3)
        return model

    def __func2__(self, input_shape):
        from keras.layers import Conv2D, ZeroPadding2D, Activation, Input
        from keras.models import Model
        from keras.layers.normalization import BatchNormalization

        x = Input(shape=input_shape)
        inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
        inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
        inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
        inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
        inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

        model = Model(inputs=x, outputs=inception_3a_5x5)
        return model

    def __func3__(self, input_shape):
        from keras.layers import Conv2D, ZeroPadding2D, Activation, Input
        from keras.models import Model
        from keras.layers.normalization import BatchNormalization
        from keras.layers.pooling import MaxPooling2D

        x = Input(shape=input_shape)
        inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
        inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
        inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
        inception_3a_pool = Activation('relu')(inception_3a_pool)
        inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

        model = Model(inputs=x, outputs=inception_3a_pool)
        return model

    def __func4__(self, input_shape):
        from keras.layers import Conv2D, Activation, Input
        from keras.models import Model
        from keras.layers.normalization import BatchNormalization

        x = Input(shape=input_shape)
        inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
        inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
        inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

        model = Model(inputs=x, outputs=inception_3a_1x1)
        return model

    def __func5__(self, input_shape):
        from keras.layers import Input, concatenate
        from keras.models import Model
        import numpy as np

        input0 = Input(shape=input_shape[0], dtype=np.float32)
        input1 = Input(shape=input_shape[1], dtype=np.float32)
        input2 = Input(shape=input_shape[2], dtype=np.float32)
        input3 = Input(shape=input_shape[3], dtype=np.float32)
        inception_3a = concatenate([input0, input1, input2, input3], axis=3)

        model = Model(inputs=[input0, input1, input2, input3], outputs=inception_3a)

        return model

    def __func6__(self, input_shape):
        from keras.layers import Conv2D, ZeroPadding2D, Activation, Input
        from keras.models import Model
        from keras.layers.normalization import BatchNormalization

        inception_3a = Input(shape=input_shape)
        inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
        inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
        inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
        inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
        inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

        model = Model(inputs=inception_3a, outputs=inception_3b_3x3)
        return model

    def __func7__(self, input_shape):
        from keras.layers import Conv2D, ZeroPadding2D, Activation, Input
        from keras.models import Model
        from keras.layers.normalization import BatchNormalization

        inception_3a  = Input(shape=input_shape)
        inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
        inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
        inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
        inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
        inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

        model = Model(inputs=inception_3a, outputs=inception_3b_5x5)
        return model

    def __func8__(self, input_shape):
        from keras.layers import Conv2D, ZeroPadding2D, Activation, Input
        from keras.models import Model
        from keras.layers.normalization import BatchNormalization
        from keras.layers.pooling import AveragePooling2D
        from keras.layers.core import Lambda
        from keras import backend as K

        inception_3a = Input(shape=input_shape)
        inception_3b_pool = Lambda(lambda x: x ** 2, name='power2_3b')(inception_3a)
        inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: x * 9, name='mult9_3b')(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
        inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
        inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
        inception_3b_pool = Activation('relu')(inception_3b_pool)
        inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

        model = Model(inputs=inception_3a, outputs=inception_3b_pool)
        return model

    def __func9__(self, input_shape):
        from keras.layers import Conv2D, Activation, Input
        from keras.models import Model
        from keras.layers.normalization import BatchNormalization

        inception_3a = Input(shape=input_shape)
        inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
        inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
        inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

        model = Model(inputs=inception_3a, outputs=inception_3b_1x1)
        return model

    def __func10__(self, input_shape):
        from keras.layers import Input, concatenate
        from keras.models import Model

        input0 = Input(shape=input_shape[0])
        input1 = Input(shape=input_shape[1])
        input2 = Input(shape=input_shape[2])
        input3 = Input(shape=input_shape[3])
        inception_3b = concatenate([input0, input1, input2, input3], axis=3)

        model = Model(inputs=[input0, input1, input2, input3], outputs=inception_3b)
        return model

    def __func11__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from Executer import utils

        inception_3b = Input(shape=input_shape)
        inception_3c_3x3 = utils.conv2d_bn(inception_3b,
                                           layer='inception_3c_3x3',
                                           cv1_out=128,
                                           cv1_filter=(1, 1),
                                           cv2_out=256,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(2, 2),
                                           padding=(1, 1))
        model = Model(inputs=inception_3b, outputs=inception_3c_3x3)
        return model

    def __func12__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from Executer import utils

        inception_3b = Input(shape=input_shape)
        inception_3c_5x5 = utils.conv2d_bn(inception_3b,
                                           layer='inception_3c_5x5',
                                           cv1_out=32,
                                           cv1_filter=(1, 1),
                                           cv2_out=64,
                                           cv2_filter=(5, 5),
                                           cv2_strides=(2, 2),
                                           padding=(2, 2))

        model = Model(inputs=inception_3b, outputs=inception_3c_5x5)
        return model

    def __func13__(self, input_shape):
        from keras.layers import ZeroPadding2D, Input
        from keras.models import Model
        from keras.layers.pooling import MaxPooling2D

        inception_3b = Input(shape=input_shape)
        inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
        inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

        model = Model(inputs=inception_3b, outputs=inception_3c_pool)
        return model

    def __func14__(self, input_shape):
        from keras.layers import Input, concatenate
        from keras.models import Model

        input0 = Input(shape=input_shape[0])
        input1 = Input(shape=input_shape[1])
        input2 = Input(shape=input_shape[2])

        inception_3c = concatenate([input0, input1, input2], axis=3)

        model = Model(inputs=[input0, input1, input2], outputs=inception_3c)
        return model

    def __func15__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from Executer import utils

        inception_3c = Input(shape=input_shape)
        inception_4a_3x3 = utils.conv2d_bn(inception_3c,
                                           layer='inception_4a_3x3',
                                           cv1_out=96,
                                           cv1_filter=(1, 1),
                                           cv2_out=192,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(1, 1),
                                           padding=(1, 1))

        model = Model(inputs=inception_3c, outputs=inception_4a_3x3)
        return model

    def __func16__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from Executer import utils

        inception_3c = Input(shape=input_shape)
        inception_4a_5x5 = utils.conv2d_bn(inception_3c,
                                           layer='inception_4a_5x5',
                                           cv1_out=32,
                                           cv1_filter=(1, 1),
                                           cv2_out=64,
                                           cv2_filter=(5, 5),
                                           cv2_strides=(1, 1),
                                           padding=(2, 2))

        model = Model(inputs=inception_3c, outputs=inception_4a_5x5)
        return model

    def __func17__(self, input_shapes):
        from keras.layers import Input
        from keras.models import Model
        from keras.layers.pooling import AveragePooling2D
        from keras.layers.core import Lambda
        from keras import backend as K
        from Executer import utils

        inception_3c = Input(shape=input_shapes)
        inception_4a_pool = Lambda(lambda x: x ** 2, name='power2_4a')(inception_3c)
        inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: x * 9, name='mult9_4a')(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)
        inception_4a_pool = utils.conv2d_bn(inception_4a_pool,
                                            layer='inception_4a_pool',
                                            cv1_out=128,
                                            cv1_filter=(1, 1),
                                            padding=(2, 2))

        model = Model(inputs=inception_3c, outputs=inception_4a_pool)
        return model

    def __func18__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from Executer import utils

        inception_3c = Input(shape=input_shape)
        inception_4a_1x1 = utils.conv2d_bn(inception_3c,
                                           layer='inception_4a_1x1',
                                           cv1_out=256,
                                           cv1_filter=(1, 1))

        model = Model(inputs=inception_3c, outputs=inception_4a_1x1)
        return model


    def __func19__(self, input_shape):
        from keras.layers import Input, concatenate
        from keras.models import Model

        inception_4a_3x3 = Input(shape=input_shape[0])
        inception_4a_5x5 = Input(shape=input_shape[1])
        inception_4a_pool = Input(shape=input_shape[2])
        inception_4a_1x1 = Input(shape=input_shape[3])
        inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

        model = Model(inputs=[inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1],
                      outputs=inception_4a)
        return model

    def __func20__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from Executer import utils

        inception_4a = Input(shape=input_shape)
        inception_4e_3x3 = utils.conv2d_bn(inception_4a,
                                           layer='inception_4e_3x3',
                                           cv1_out=160,
                                           cv1_filter=(1, 1),
                                           cv2_out=256,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(2, 2),
                                           padding=(1, 1))

        model = Model(inputs=inception_4a, outputs=inception_4e_3x3)
        return model

    def __func21__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from Executer import utils

        inception_4a = Input(shape=input_shape)
        inception_4e_5x5 = utils.conv2d_bn(inception_4a,
                                           layer='inception_4e_5x5',
                                           cv1_out=64,
                                           cv1_filter=(1, 1),
                                           cv2_out=128,
                                           cv2_filter=(5, 5),
                                           cv2_strides=(2, 2),
                                           padding=(2, 2))

        model = Model(inputs=inception_4a, outputs=inception_4e_5x5)
        return model

    def __func22__(self, input_shape):
        from keras.layers import ZeroPadding2D, Input
        from keras.models import Model
        from keras.layers.pooling import MaxPooling2D

        inception_4a = Input(shape=input_shape)
        inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
        inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

        model = Model(inputs=inception_4a, outputs=inception_4e_pool)
        return model

    def __func23__(self, input_shape):
        from keras.layers import Input, concatenate
        from keras.models import Model

        inception_4e_3x3 = Input(shape=input_shape[0])
        inception_4e_5x5 = Input(shape=input_shape[1])
        inception_4e_pool = Input(shape=input_shape[2])
        inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

        model = Model(inputs=[inception_4e_3x3, inception_4e_5x5, inception_4e_pool], outputs=inception_4e)
        return model

    def __func24__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from Executer import utils

        inception_4e = Input(shape=input_shape)
        inception_5a_3x3 = utils.conv2d_bn(inception_4e,
                                           layer='inception_5a_3x3',
                                           cv1_out=96,
                                           cv1_filter=(1, 1),
                                           cv2_out=384,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(1, 1),
                                           padding=(1, 1))

        model = Model(inputs=inception_4e, outputs=inception_5a_3x3)
        return model

    def __func25__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from keras.layers.pooling import AveragePooling2D
        from keras.layers.core import Lambda
        from keras import backend as K
        from Executer import utils

        inception_4e = Input(shape=input_shape)
        inception_5a_pool = Lambda(lambda x: x ** 2, name='power2_5a')(inception_4e)
        inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: x * 9, name='mult9_5a')(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)
        inception_5a_pool = utils.conv2d_bn(inception_5a_pool,
                                            layer='inception_5a_pool',
                                            cv1_out=96,
                                            cv1_filter=(1, 1),
                                            padding=(1, 1))
        model = Model(inputs=inception_4e, outputs=inception_5a_pool)
        return model

    def __func26__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from Executer import utils

        inception_4e = Input(shape=input_shape)

        inception_5a_1x1 = utils.conv2d_bn(inception_4e,
                                           layer='inception_5a_1x1',
                                           cv1_out=256,
                                           cv1_filter=(1, 1))
        model = Model(inputs=inception_4e, outputs=inception_5a_1x1)
        return model

    def __func27__(self, input_shape):
        from keras.layers import Input, concatenate
        from keras.models import Model

        inception_5a_3x3 = Input(shape=input_shape[0])
        inception_5a_pool = Input(shape=input_shape[1])
        inception_5a_1x1 = Input(shape=input_shape[2])

        inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)
        model = Model(inputs=[inception_5a_3x3, inception_5a_pool, inception_5a_1x1], outputs=inception_5a)
        return model

    def __func28__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from Executer import utils

        inception_5a = Input(shape=input_shape)
        inception_5b_3x3 = utils.conv2d_bn(inception_5a,
                                           layer='inception_5b_3x3',
                                           cv1_out=96,
                                           cv1_filter=(1, 1),
                                           cv2_out=384,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(1, 1),
                                           padding=(1, 1))

        model = Model(inputs=inception_5a, outputs=inception_5b_3x3)
        return model

    def __func29__(self, input_shape):
        from keras.layers import ZeroPadding2D, Input
        from keras.models import Model
        from keras.layers.pooling import MaxPooling2D
        from Executer import utils

        inception_5a = Input(shape=input_shape)
        inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
        inception_5b_pool = utils.conv2d_bn(inception_5b_pool,
                                            layer='inception_5b_pool',
                                            cv1_out=96,
                                            cv1_filter=(1, 1))
        inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

        model = Model(inputs=inception_5a, outputs=inception_5b_pool)
        return model

    def __func30__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from Executer import utils

        inception_5a = Input(shape=input_shape)
        inception_5b_1x1 = utils.conv2d_bn(inception_5a,
                                           layer='inception_5b_1x1',
                                           cv1_out=256,
                                           cv1_filter=(1, 1))
        model = Model(inputs=inception_5a, outputs=inception_5b_1x1)
        return model

    def __func31__(self, input_shape):
        from keras.layers import Input, concatenate
        from keras.models import Model

        inception_5b_3x3 = Input(shape=input_shape[0])
        inception_5b_pool = Input(shape=input_shape[1])
        inception_5b_1x1 = Input(shape=input_shape[2])
        inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

        model = Model(inputs=[inception_5b_3x3, inception_5b_pool, inception_5b_1x1], outputs=inception_5b)
        return model

    def __func32__(self, input_shape):
        from keras.layers import Input
        from keras.models import Model
        from keras.layers.pooling import AveragePooling2D
        from keras.layers.core import Lambda, Flatten, Dense
        from keras import backend as K

        inception_5b = Input(shape=input_shape)
        av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
        reshape_layer = Flatten()(av_pool)
        dense_layer = Dense(128, name='dense_layer')(reshape_layer)
        norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

        model = Model(inputs=inception_5b, outputs=norm_layer)
        return model

    def __init__(self):
        self.operations = []
        weights_dict = utils.load_weights()

        operation0 = operation(0, self.__func0__, (96, 96, 3), weights_dict)
        operation1 = operation(1, self.__func1__, (12, 12, 192), weights_dict)
        operation2 = operation(2, self.__func2__, (12, 12, 192), weights_dict)
        operation3 = operation(3, self.__func3__, (12, 12, 192), weights_dict)
        operation4 = operation(4, self.__func4__, (12, 12, 192), weights_dict)
        operation5 = operation(5, self.__func5__, [(12, 12, 128), (12, 12, 32), (12, 12, 32), (12, 12, 64)],
                               weights_dict)
        operation6 = operation(6, self.__func6__, (12, 12, 256), weights_dict)
        operation7 = operation(7, self.__func7__, (12, 12, 256), weights_dict)
        operation8 = operation(8, self.__func8__, (12, 12, 256), weights_dict)
        operation9 = operation(9, self.__func9__, (12, 12, 256), weights_dict)
        operation10 = operation(10, self.__func10__, [(12, 12, 128), (12, 12, 64), (12, 12, 64), (12, 12, 64)],
                               weights_dict)
        operation11 = operation(11, self.__func11__, (12, 12, 320), weights_dict)
        operation12 = operation(12, self.__func12__, (12, 12, 320), weights_dict)
        operation13 = operation(13, self.__func13__, (12, 12, 320), weights_dict)
        operation14 = operation(14, self.__func14__, [(6, 6, 256), (6, 6, 64), (6, 6, 320)],
                                weights_dict)
        operation15 = operation(15, self.__func15__, (6, 6, 640), weights_dict)
        operation16 = operation(16, self.__func16__, (6, 6, 640), weights_dict)
        operation17 = operation(17, self.__func17__, (6, 6, 640), weights_dict)
        operation18 = operation(18, self.__func18__, (6, 6, 640), weights_dict)
        operation19 = operation(19, self.__func19__, [(6, 6, 192), (6, 6, 64), (6, 6, 128), (6, 6, 256)],
                                weights_dict)
        operation20 = operation(20, self.__func20__, (6, 6, 640), weights_dict)
        operation21 = operation(21, self.__func21__, (6, 6, 640), weights_dict)
        operation22 = operation(22, self.__func22__, (6, 6, 640), weights_dict)
        operation23 = operation(23, self.__func23__, [(3, 3, 256), (3, 3, 128), (3, 3, 640)],
                                weights_dict)
        operation24 = operation(24, self.__func24__, (3, 3, 1024), weights_dict)
        operation25 = operation(25, self.__func25__, (3, 3, 1024), weights_dict)
        operation26 = operation(26, self.__func26__, (3, 3, 1024), weights_dict)
        operation27 = operation(27, self.__func27__, [(3, 3, 384), (3, 3, 96), (3, 3, 256)],
                                weights_dict)
        operation28 = operation(28, self.__func28__, (3, 3, 736), weights_dict)
        operation29 = operation(29, self.__func29__, (3, 3, 736), weights_dict)
        operation30 = operation(30, self.__func30__, (3, 3, 736), weights_dict)
        operation31 = operation(31, self.__func31__, [(3, 3, 384), (3, 3, 96), (3, 3, 256)],
                                weights_dict)
        operation32 = operation(32, self.__func32__, (3, 3, 736), weights_dict)

        self.operations.append(operation0)
        self.operations.append(operation1)
        self.operations.append(operation2)
        self.operations.append(operation3)
        self.operations.append(operation4)
        self.operations.append(operation5)
        self.operations.append(operation6)
        self.operations.append(operation7)
        self.operations.append(operation8)
        self.operations.append(operation9)
        self.operations.append(operation10)
        self.operations.append(operation11)
        self.operations.append(operation12)
        self.operations.append(operation13)
        self.operations.append(operation14)
        self.operations.append(operation15)
        self.operations.append(operation16)
        self.operations.append(operation17)
        self.operations.append(operation18)
        self.operations.append(operation19)
        self.operations.append(operation20)
        self.operations.append(operation21)
        self.operations.append(operation22)
        self.operations.append(operation23)
        self.operations.append(operation24)
        self.operations.append(operation25)
        self.operations.append(operation26)
        self.operations.append(operation27)
        self.operations.append(operation28)
        self.operations.append(operation29)
        self.operations.append(operation30)
        self.operations.append(operation31)
        self.operations.append(operation32)

    def excute(self, operationid, inputdata):
        return self.operations[operationid].excute(inputdata)















































