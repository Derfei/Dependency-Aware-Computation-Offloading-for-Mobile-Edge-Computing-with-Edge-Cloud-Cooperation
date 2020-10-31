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
@description: vgg16的执行
'''
class operation:

    def __init__(self, operation_id, generate_operation_model, input_shape,weights_model):
        import numpy as np

        self.operation_id = operation_id
        self.operation_model = generate_operation_model(input_shape)
        self.input_shape = input_shape

        'load the weight'
        for layer in self.operation_model.layers:
            try:
                if weights_model.get_layer(layer.name) != None:
                    layer.set_weights(weights_model.get_layer(name=layer.name).get_weights())
            except Exception as e:
                print("cannot find the layer {0} in the vgg model and exception is {1}".format(layer.name,
                                                                                               e))
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
        print("the operation {0} output shape is {1}".format(self.operation_id, np.shape(embedding)))
        return embedding
        pass

class excuteVgg16_sa:

    def __func0__(self, input_shape):
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D

        img_input = Input(shape=input_shape)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        model = Model(inputs=img_input, outputs=x)
        return model

    def __func1__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.models import Model


        input = Input(shape=input_shape)
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(input)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        model = Model(inputs=input, outputs=x)
        return model

    def __func2__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.models import Model

        input = Input(shape=input_shape)
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(input)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        model = Model(inputs=input, outputs=x)
        return model

    # def __func3__(self, input_shape):
    #     from keras.layers import Input
    #     from keras.layers import Conv2D
    #     from keras.layers import MaxPooling2D
    #     from keras.models import Model
    #
    #     input = Input(shape=input_shape)
    #     # Block 4
    #     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(input)
    #     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    #     x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    #     x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #
    #     model = Model(inputs=input, outputs=x)
    #     return model

    def __func3__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.models import Model

        input = Input(shape=input_shape)
        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(input)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        model = Model(inputs=input, outputs=x)
        return model

    def __func4__(self, input_shape):
        from keras.layers import Flatten
        from keras.layers import Dense
        from keras.layers import Input
        from keras.models import Model

        input = Input(shape=input_shape)
        # Classification block
        x = Flatten(name='flatten')(input)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

        model = Model(inputs=input, outputs=x)
        return model


    def __init__(self):
        from Executer.vgg16 import  vgg16
        self.operations = []
        weights_model = vgg16(input_shape=(224,224, 3),
                              classes=1000).model

        operation0 = operation(0, self.__func0__, (224, 224, 3), weights_model)
        operation1 = operation(1, self.__func1__, (112, 112, 64), weights_model)
        operation2 = operation(2, self.__func2__, (56, 56, 128), weights_model)
        operation3 = operation(3, self.__func3__, (14, 14, 512), weights_model)
        operation4 = operation(4, self.__func4__, (7, 7, 512), weights_model)

        self.operations.append(operation0)
        self.operations.append(operation1)
        self.operations.append(operation2)
        self.operations.append(operation3)
        self.operations.append(operation4)


    def excute(self, operationid, inputdata):
        return self.operations[operationid].excute(inputdata)


