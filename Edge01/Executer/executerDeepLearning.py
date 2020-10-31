# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description: 深度学习任务的执行
'''
from Executer import utils
class operation:

    def __init__(self, operation_id, generate_operation_model, input_shape, weights_dict):
        import numpy as np

        self.operation_id = operation_id
        self.operation_model = generate_operation_model(input_shape)
        self.input_shape = input_shape

        'load the weight'
        weights = utils.weights
        # weights_dict = utils.load_weights()

        for name in weights:
            try:
                if self.operation_model.get_layer(name) != None:
                    self.operation_model.get_layer(name).set_weights(weights_dict[name])
            except Exception as e:
                pass
        self.operation_model.predict(np.array([np.zeros(shape=input_shape, dtype=np.float32)]))

    def excute(self, input):
        import numpy as np

        x_input = np.array(input)
        if x_input.shape[0] == self.input_shape[0] and x_input.shape[1] == self.input_shape[1]:
            x_input = np.array([x_input])
        print("the operation {0} input shape is:{1}".format(self.operation_id, x_input.shape))
        embedding = self.operation_model.predict_on_batch(x_input)
        return embedding

class excuterDeepLearning:

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

        model = Model(inputs=[myInput], outputs=x)
        return model

    def __func_3a__(self, input_shape):
        from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
        from keras.models import Model
        from keras.layers.normalization import BatchNormalization
        from keras.layers.pooling import MaxPooling2D

        x = Input(shape=input_shape)
        inception_3a_3x3 = Conv2D(96, (1, 1), name='inception_3a_3x3_conv1')(x)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn1')(inception_3a_3x3)
        inception_3a_3x3 = Activation('relu')(inception_3a_3x3)
        inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
        inception_3a_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(inception_3a_3x3)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_3x3_bn2')(inception_3a_3x3)
        inception_3a_3x3 = Activation('relu')(inception_3a_3x3)

        inception_3a_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(x)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn1')(inception_3a_5x5)
        inception_3a_5x5 = Activation('relu')(inception_3a_5x5)
        inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
        inception_3a_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(inception_3a_5x5)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_5x5_bn2')(inception_3a_5x5)
        inception_3a_5x5 = Activation('relu')(inception_3a_5x5)

        inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
        inception_3a_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(inception_3a_pool)
        inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_pool_bn')(inception_3a_pool)
        inception_3a_pool = Activation('relu')(inception_3a_pool)
        inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)

        inception_3a_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(x)
        inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3a_1x1_bn')(inception_3a_1x1)
        inception_3a_1x1 = Activation('relu')(inception_3a_1x1)

        inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, inception_3a_pool, inception_3a_1x1], axis=3)

        model = Model(inputs=x, outputs=inception_3a)
        return model

    def __func_3b__(self, input_shape):
        from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
        from keras.models import Model
        from keras.layers.normalization import BatchNormalization
        from keras.layers.pooling import AveragePooling2D
        from keras.layers.core import Lambda
        from keras import backend as K

        inception_3a = Input(shape=input_shape)
        inception_3b_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception_3a)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn1')(inception_3b_3x3)
        inception_3b_3x3 = Activation('relu')(inception_3b_3x3)
        inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
        inception_3b_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(inception_3b_3x3)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_3x3_bn2')(inception_3b_3x3)
        inception_3b_3x3 = Activation('relu')(inception_3b_3x3)

        inception_3b_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception_3a)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn1')(inception_3b_5x5)
        inception_3b_5x5 = Activation('relu')(inception_3b_5x5)
        inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
        inception_3b_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(inception_3b_5x5)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_5x5_bn2')(inception_3b_5x5)
        inception_3b_5x5 = Activation('relu')(inception_3b_5x5)

        inception_3b_pool = Lambda(lambda x: x ** 2, name='power2_3b')(inception_3a)
        inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: x * 9, name='mult9_3b')(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_3b')(inception_3b_pool)
        inception_3b_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(inception_3b_pool)
        inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_pool_bn')(inception_3b_pool)
        inception_3b_pool = Activation('relu')(inception_3b_pool)
        inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

        inception_3b_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception_3a)
        inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='inception_3b_1x1_bn')(inception_3b_1x1)
        inception_3b_1x1 = Activation('relu')(inception_3b_1x1)

        inception_3b = concatenate([inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3)

        model = Model(inputs=inception_3a, outputs=inception_3b)
        return model

    def __func_3c__(self, input_shape):
        from keras.layers import ZeroPadding2D, Input, concatenate
        from keras.models import Model
        from keras.layers.pooling import MaxPooling2D

        inception_3b = Input(shape=input_shape)

        # Inception3c
        inception_3c_3x3 = utils.conv2d_bn(inception_3b,
                                           layer='inception_3c_3x3',
                                           cv1_out=128,
                                           cv1_filter=(1, 1),
                                           cv2_out=256,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(2, 2),
                                           padding=(1, 1))

        inception_3c_5x5 = utils.conv2d_bn(inception_3b,
                                           layer='inception_3c_5x5',
                                           cv1_out=32,
                                           cv1_filter=(1, 1),
                                           cv2_out=64,
                                           cv2_filter=(5, 5),
                                           cv2_strides=(2, 2),
                                           padding=(2, 2))

        inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
        inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

        inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)
        model = Model(inputs=inception_3b, outputs=inception_3c)

        return model

    def __func_4a__(self, input_shape):
        from keras.layers import Input, concatenate
        from keras.models import Model
        from keras.layers.pooling import AveragePooling2D
        from keras.layers.core import Lambda
        from keras import backend as K

        inception_3c = Input(shape=input_shape)
        # inception 4a
        inception_4a_3x3 = utils.conv2d_bn(inception_3c,
                                           layer='inception_4a_3x3',
                                           cv1_out=96,
                                           cv1_filter=(1, 1),
                                           cv2_out=192,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(1, 1),
                                           padding=(1, 1))
        inception_4a_5x5 = utils.conv2d_bn(inception_3c,
                                           layer='inception_4a_5x5',
                                           cv1_out=32,
                                           cv1_filter=(1, 1),
                                           cv2_out=64,
                                           cv2_filter=(5, 5),
                                           cv2_strides=(1, 1),
                                           padding=(2, 2))

        inception_4a_pool = Lambda(lambda x: x ** 2, name='power2_4a')(inception_3c)
        inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: x * 9, name='mult9_4a')(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_4a')(inception_4a_pool)
        inception_4a_pool = utils.conv2d_bn(inception_4a_pool,
                                            layer='inception_4a_pool',
                                            cv1_out=128,
                                            cv1_filter=(1, 1),
                                            padding=(2, 2))
        inception_4a_1x1 = utils.conv2d_bn(inception_3c,
                                           layer='inception_4a_1x1',
                                           cv1_out=256,
                                           cv1_filter=(1, 1))
        inception_4a = concatenate([inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3)

        model  = Model(inputs=inception_3c, outputs=inception_4a)

        return model

    def __func_4e__(self, input_shape):
        from keras.layers import ZeroPadding2D, Input, concatenate
        from keras.models import Model
        from keras.layers.pooling import MaxPooling2D

        inception_4a = Input(input_shape)

        # inception4e
        inception_4e_3x3 = utils.conv2d_bn(inception_4a,
                                           layer='inception_4e_3x3',
                                           cv1_out=160,
                                           cv1_filter=(1, 1),
                                           cv2_out=256,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(2, 2),
                                           padding=(1, 1))
        inception_4e_5x5 = utils.conv2d_bn(inception_4a,
                                           layer='inception_4e_5x5',
                                           cv1_out=64,
                                           cv1_filter=(1, 1),
                                           cv2_out=128,
                                           cv2_filter=(5, 5),
                                           cv2_strides=(2, 2),
                                           padding=(2, 2))
        inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
        inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

        inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

        model = Model(inputs=inception_4a, outputs=inception_4e)

        return model

    def __func_5a__(self, input_shape):
        from keras.layers import Input, concatenate
        from keras.models import Model
        from keras.layers.pooling import AveragePooling2D
        from keras.layers.core import Lambda
        from keras import backend as K

        inception_4e = Input(shape=input_shape)
        # inception5a
        inception_5a_3x3 = utils.conv2d_bn(inception_4e,
                                           layer='inception_5a_3x3',
                                           cv1_out=96,
                                           cv1_filter=(1, 1),
                                           cv2_out=384,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(1, 1),
                                           padding=(1, 1))

        inception_5a_pool = Lambda(lambda x: x ** 2, name='power2_5a')(inception_4e)
        inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: x * 9, name='mult9_5a')(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: K.sqrt(x), name='sqrt_5a')(inception_5a_pool)
        inception_5a_pool = utils.conv2d_bn(inception_5a_pool,
                                            layer='inception_5a_pool',
                                            cv1_out=96,
                                            cv1_filter=(1, 1),
                                            padding=(1, 1))
        inception_5a_1x1 = utils.conv2d_bn(inception_4e,
                                           layer='inception_5a_1x1',
                                           cv1_out=256,
                                           cv1_filter=(1, 1))

        inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

        model = Model(inputs=inception_4e, outputs=inception_5a)

        return model

    def __func_5b__(self, input_shape):
        from keras.layers import ZeroPadding2D, Input, concatenate
        from keras.models import Model
        from keras.layers.pooling import MaxPooling2D

        inception_5a = Input(input_shape)

        # inception_5b
        inception_5b_3x3 = utils.conv2d_bn(inception_5a,
                                           layer='inception_5b_3x3',
                                           cv1_out=96,
                                           cv1_filter=(1, 1),
                                           cv2_out=384,
                                           cv2_filter=(3, 3),
                                           cv2_strides=(1, 1),
                                           padding=(1, 1))
        inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)
        inception_5b_pool = utils.conv2d_bn(inception_5b_pool,
                                            layer='inception_5b_pool',
                                            cv1_out=96,
                                            cv1_filter=(1, 1))
        inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

        inception_5b_1x1 = utils.conv2d_bn(inception_5a,
                                           layer='inception_5b_1x1',
                                           cv1_out=256,
                                           cv1_filter=(1, 1))
        inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

        model = Model(inputs=inception_5a, outputs=inception_5b)
        return model

    def __func_last__(self, input_shape):
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

        return  model


    def __init__(self):
        weights_dict = utils.load_weights()
        self.operations = []
        print("begin to init the operation")
        operation0 = operation(0, self.__func0__, (96, 96, 3), weights_dict)
        print("end to init the operation 0")
        operation1 = operation(1, self.__func_3a__, (12, 12, 192), weights_dict)
        operation2 = operation(2, self.__func_3b__, (12, 12, 256), weights_dict)
        operation3 = operation(3, self.__func_3c__, (12, 12, 320), weights_dict)
        operation4 = operation(4, self.__func_4a__, (6, 6, 640), weights_dict)
        operation5 = operation(5, self.__func_4e__, (6, 6, 640), weights_dict)
        operation6 = operation(6, self.__func_5a__, (3, 3, 1024), weights_dict)
        operation7 = operation(7, self.__func_5b__, (3, 3, 736), weights_dict)
        operation8 = operation(8, self.__func_last__, (3, 3, 736), weights_dict)
        print("end to init hte operation")

        self.operations.append(operation0)
        self.operations.append(operation1)
        self.operations.append(operation2)
        self.operations.append(operation3)
        self.operations.append(operation4)
        self.operations.append(operation5)
        self.operations.append(operation6)
        self.operations.append(operation7)
        self.operations.append(operation8)

    def excute(self, operationid, inputdata):
        if operationid == 0:
            return  self.operations[operationid].excute(inputdata)
        else:
            return self.operations[operationid].excute(inputdata[0])




















