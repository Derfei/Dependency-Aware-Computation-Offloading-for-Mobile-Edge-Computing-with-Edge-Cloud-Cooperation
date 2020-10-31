# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description: 使用残差网络进行实验
'''
class operation:

    def __init__(self, operation_id, generate_operation_model, input_shape,weights_model):
        import numpy as np

        self.operation_id = operation_id
        self.operation_model = generate_operation_model(input_shape)
        self.input_shape = input_shape

        # 'load the weight'
        # for layer in self.operation_model.layers:
        #     try:
        #         if weights_model.get_layer(layer.name) != None:
        #             layer.set_weights(weights_model.get_layer(name=layer.name).get_weights())
        #     except Exception as e:
        #         print("cannot find the layer {0} in the vgg model and exception is {1}".format(layer.name,
        #                                                                                        e))
        #         pass
        # if type(input_shape) == list:
        #     testdata = []
        #     for tmp in input_shape:
        #         testdata.append([np.zeros(shape=tmp, dtype=np.float32)])
        #     self.operation_model.predict(testdata)
        #     pass
        # else:
        #     self.operation_model.predict(np.array([np.zeros(shape=input_shape, dtype=np.float32)]))

    def excute(self, input):
        import numpy as np

        x_input = input
        # if np.shape(input)[0] == self.input_shape[0]:
        #     x_input = [input]
        if type(self.input_shape) != list:
            x_input = np.array(x_input)

        if type(self.input_shape) == list:
            input_data = []
            print("the raw shape of the input is {0} of operation {1}".format(np.shape(input),
                                                                              self.operation_id))
            for i in range(len(self.input_shape)):
                print("operation {0} the input shape is {1}".format(self.operation_id,
                                                                    np.shape(input[i])))
                # x_input.append(np.array(input[i]))
                input_data.append(input[i])
                print("operation {0} the input shape is {1}".format(self.operation_id,
                                                                    np.shape(input_data[i])))

            print("the operation {0} input shape is:{1}".format(self.operation_id, np.shape(x_input)))
            embedding = self.operation_model.predict(input_data)
            return embedding
        print("the operation {0} input shape is:{1}".format(self.operation_id, np.shape(x_input)))
        embedding = self.operation_model.predict(x_input)
        print("the operation {0} output shape is:{1}".format(self.operation_id, np.shape(embedding)))
        return embedding


class excuteResnet50Onetask:

    def __func0__(self, input_shape):
        from Executer.resnet50 import resnet50
        from keras.layers import Input
        from keras.layers import Dense
        from keras.layers import Activation
        from keras.layers import Flatten
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.layers import ZeroPadding2D
        from keras.layers import AveragePooling2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        img_input = Input(shape=input_shape)
        x = ZeroPadding2D((3, 3))(img_input)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = resnet50.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = resnet50.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = resnet50.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = resnet50.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = resnet50.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = resnet50.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = resnet50.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = resnet50.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = resnet50.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = resnet50.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = resnet50.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)

        x = Flatten()(x)
        x = Dense(1000, activation='softmax', name='fc1000')(x)

        model = Model(inputs=img_input, outputs=x)
        return model


    def __init__(self):
        from Executer.resnet50 import resnet50
        self.operations = []
        weights_model = resnet50(input_shape=(224, 224, 3)).model

        operation0 = operation(0, self.__func0__, (224, 224, 3), weights_model)

        self.operations.append(operation0)



    def excute(self, operationid, inputdata):
        return self.operations[operationid].excute(inputdata)