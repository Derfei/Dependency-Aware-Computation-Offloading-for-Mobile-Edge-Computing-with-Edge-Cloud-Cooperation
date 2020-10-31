# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
class operation:

    def __init__(self, operation_id, generate_operation_model, input_shape,weights_model_vgg16, weights_model_vgg19):
        import numpy as np

        self.operation_id = operation_id
        self.operation_model = generate_operation_model(input_shape)
        self.input_shape = input_shape


        'load the weight'
        for layer in self.operation_model.layers:
            try:
                if weights_model_vgg16.get_layer(layer.name) != None:
                    layer.set_weights(weights_model_vgg16.get_layer(layer.name).get_weights())
                    continue
                if weights_model_vgg19.get_layer(layer.name) != None:
                    layer.set_weights(weights_model_vgg19.get_layer(layer.name).get_weights())
                    continue

            except Exception as e:
                print("not find the weight of layer {0} and error is {1}".format(layer.name, e))
                pass

        if self.operation_id != 0:
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

        if self.operation_id == 0:
            return input


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
        return embedding
        pass

class excuteVggboostVggCMS:

    # (224, 224, 3)
    def __func0__(self, input_shape):
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Dense
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D

        img_input = Input(shape=input_shape)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='16_block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='16_block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='16_block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='16_block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='16_block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='16_block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='16_block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='16_block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='16_block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='16_block3_pool')(x)

        model = Model(inputs=img_input, outputs=x)
        return model

    # (224, 224, 3)
    def __func1__(self, input_shape):
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D

        img_input = Input(shape=input_shape)
        # vgg19_input = Input(shape=(224, 224, 3))
        # Block 1
        x_vgg19 = Conv2D(64, (3, 3), activation='relu', padding='same', name='19_block1_conv1')(img_input)
        x_vgg19 = Conv2D(64, (3, 3), activation='relu', padding='same', name='19_block1_conv2')(x_vgg19)
        x_vgg19 = MaxPooling2D((2, 2), strides=(2, 2), name='19_block1_pool')(x_vgg19)

        # Block 2
        x_vgg19 = Conv2D(128, (3, 3), activation='relu', padding='same', name='19_block2_conv1')(x_vgg19)
        x_vgg19 = Conv2D(128, (3, 3), activation='relu', padding='same', name='19_block2_conv2')(x_vgg19)
        x_vgg19 = MaxPooling2D((2, 2), strides=(2, 2), name='19_block2_pool')(x_vgg19)

        # Block 3
        x_vgg19 = Conv2D(256, (3, 3), activation='relu', padding='same', name='19_block3_conv1')(x_vgg19)
        x_vgg19 = Conv2D(256, (3, 3), activation='relu', padding='same', name='19_block3_conv2')(x_vgg19)
        x_vgg19 = Conv2D(256, (3, 3), activation='relu', padding='same', name='19_block3_conv3')(x_vgg19)
        x_vgg19 = Conv2D(256, (3, 3), activation='relu', padding='same', name='19_block3_conv4')(x_vgg19)
        x_vgg19 = MaxPooling2D((2, 2), strides=(2, 2), name='19_block3_pool')(x_vgg19)

        model = Model(inputs=img_input, outputs=x_vgg19)
        return model


    #
    def __func2__(self, input_shape):
        from keras.models import Model
        from keras.layers import Dense
        from keras.layers import Input
        from keras import layers

        x_vgg19 = Input(shape=input_shape[0])
        x = Input(shape=input_shape[1])


        output = layers.add([x_vgg19, x])
        output = Dense(1000, activation='softmax', name='predictions')(output)

        model = Model(inputs=[x_vgg19, x], outputs=output)
        return model

    def __init__(self):
        from Executer.vgg16boostvgg19 import util_vgg16boostvgg19
        import gc
        self.operations = []
        weights_model_vgg16 = util_vgg16boostvgg19.get_vgg16model()
        weights_model_vgg19 = util_vgg16boostvgg19.get_vgg19model()

        operation0 = operation(0, self.__func0__, (224, 224, 3), weights_model_vgg16 ,weights_model_vgg19)
        operation1 = operation(1, self.__func1__, (224, 224, 3), weights_model_vgg16, weights_model_vgg19)
        operation2 = operation(2, self.__func2__, [(28, 28, 256,), (28, 28, 256)], weights_model_vgg16, weights_model_vgg19)

        self.operations.append(operation0)
        self.operations.append(operation1)
        self.operations.append(operation2)

    def excute(self, operationid, inputdata):
        return self.operations[operationid].excute(inputdata)
