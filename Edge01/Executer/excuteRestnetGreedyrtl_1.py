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
            print("the operation {0} output shape is:{1}".format(self.operation_id, np.shape(embedding)))
            return embedding
        print("the operation {0} input shape is:{1}".format(self.operation_id, np.shape(x_input)))
        embedding = self.operation_model.predict(x_input)
        print("the operation {0} output shape is:{1}".format(self.operation_id, np.shape(embedding)))
        return embedding


class excuteResnet50Greedyrtl_1:

    def __func0__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.layers import ZeroPadding2D
        from keras.layers import BatchNormalization
        from keras.models import Model

        # block 0
        img_input = Input(shape=input_shape)
        x = ZeroPadding2D((3, 3))(img_input)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        model = Model(inputs=img_input, outputs=x)
        return model

    # 55  55 64
    def __func1__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        from keras import layers
        import keras.backend as K

        input = Input(shape=input_shape)
        # block 1 conv_block input_tensor, kernel_size, filters, stage, block, strides=(2, 2)
        input_tensor = input
        kernel_size = 3
        filters = [64, 64, 256]
        stage = 2
        block = 'a'
        strides = (1, 1)
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        input_tensor = input
        kernel_size = 3
        filters = [64, 64, 256]
        stage = 2
        block = 'a'
        strides = (1, 1)
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        a = layers.add([x, shortcut])
        a = Activation('relu')(a)

        model = Model(inputs=input, outputs=a)
        return model

    # 55 55 256
    def __func2__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        import keras.backend as K

        input = Input(shape=input_shape)
        input_tensor = input
        kernel_size = 3
        filters = [64, 64, 256]
        stage = 2
        block = 'b'
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        model = Model(inputs=input, outputs=x)
        return model

    # (55, 55, 256), (55, 55, 256)]
    def __func3__(self, input_shape):
        from keras.layers import Input
        from keras import layers
        from keras.layers import Activation
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        import keras.backend as K

        input = Input(shape=input_shape[0])
        input_tensor = Input(shape=input_shape[1])

        # block 5
        x = layers.add([input, input_tensor])
        x = Activation('relu')(x)


        kernel_size = 3
        filters = [64, 64, 256]
        stage = 2
        block = 'c'
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x_1 = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(x)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x_1)
        x_1 = Activation('relu')(x_1)

        x_1 = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x_1)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x_1)
        x_1 = Activation('relu')(x_1)

        x_1 = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x_1)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x_1)

        # block 7
        x = layers.add([x_1, x])
        x = Activation('relu')(x)

        model = Model(inputs=[input, input_tensor], outputs=x)
        return model

    # (5, 5, 256)
    def __func4__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        import keras.backend as K
        from keras import layers
        from keras.layers import Activation
        from keras.models import Model

        input = Input(shape=input_shape)
        # block 8 conv_block input_tensor, kernel_size, filters, stage, block, strides=(2, 2)
        input_tensor = input
        kernel_size = 3
        filters = [128, 128, 512]
        stage = 3
        block = 'a'
        strides = (2, 2)
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        kernel_size = 3
        filters = [128, 128, 512]
        stage = 3
        block = 'a'
        strides = (2, 2)
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        # block 9
        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        # block 10
        x = layers.add([x, shortcut])
        x = Activation('relu')(x)

        model = Model(inputs=input, outputs=x)
        return model

    # (28, 28, 512)
    def __func5__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        import keras.backend as K

        input = Input(shape=input_shape)
        # block 11 identify_block input_tensor, kernel_size, filters, stage, block
        input_tensor = input
        kernel_size = 3
        filters = [128, 128, 512]
        stage = 3
        block = 'b'
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        model = Model(inputs=input, outputs=x)
        return model

    # [(28, 28, 512), (28, 28, 512)]
    def __func6__(self, input_shape):
        from keras.layers import Input
        from keras import layers
        from keras.layers import Activation
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        import keras.backend as K

        input = Input(shape=input_shape[0])
        input_tensor = Input(shape=input_shape[1])
        # block 12
        x = layers.add([input, input_tensor])
        x = Activation('relu')(x)

        # block 13
        kernel_size = 3
        filters = [128, 128, 512]
        stage = 3
        block = 'c'
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x_1 = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(x)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x_1)
        x_1 = Activation('relu')(x_1)

        x_1 = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x_1)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x_1)
        x_1 = Activation('relu')(x_1)

        x_1 = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x_1)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x_1)

        # block 14
        x_a = layers.add([x_1, x])
        x_a = Activation('relu')(x_a)

        # blok 15
        kernel_size = 3
        filters = [128, 128, 512]
        stage = 3
        block = 'd'
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x_2 = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(x_a)
        x_2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x_2)
        x_2 = Activation('relu')(x_2)

        x_2 = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x_2)
        x_2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x_2)
        x_2 = Activation('relu')(x_2)

        x_2 = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x_2)
        x_2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x_2)

        # block 16
        a = layers.add([x_2, x_a])
        a = Activation('relu')(a)

        model = Model(inputs=[input, input_tensor], outputs=a)
        return model

    # (28, 28, 512)
    def __func7__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        from keras.layers import Input
        from keras import layers
        from keras.layers import Activation
        from keras.models import Model
        import keras.backend as K

        input = Input(shape=input_shape)
        # block 17 conv_block input_tensor, kernel_size, filters, stage, block, strides=(2, 2)
        input_tensor = input
        kernel_size = 3
        filters = [256, 256, 1024]
        stage = 4
        block = 'a'
        strides = (2, 2)
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        # block 18
        kernel_size = 3
        filters = [256, 256, 1024]
        stage = 4
        block = 'a'
        strides = (2, 2)
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        # block 19
        x = layers.add([x, shortcut])
        x = Activation('relu')(x)

        # block 20
        kernel_size = 3
        filters = [256, 256, 1024]
        stage = 4
        block = 'b'
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x_1 = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(x)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x_1)
        x_1 = Activation('relu')(x_1)

        x_1 = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x_1)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x_1)
        x_1 = Activation('relu')(x_1)


        x_1 = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x_1)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x_1)

        # block 21
        x = layers.add([x_1, x])
        x = Activation('relu')(x)

        # block 22
        kernel_size = 3
        filters = [256, 256, 1024]
        stage = 4
        block = 'c'
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x_2 = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(x)
        x_2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x_2)
        x_2 = Activation('relu')(x_2)

        x_2 = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x_2)
        x_2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x_2)
        x_2 = Activation('relu')(x_2)

        x_2 = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x_2)
        x_2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x_2)

        # block 23
        # block 23
        x = layers.add([x_2, x])
        x = Activation('relu')(x)

        # bloack 24
        # block 24 identify_block input_tensor, kernel_size, filters, stage, block
        kernel_size = 3
        filters = [256, 256, 1024]
        stage = 4
        block = 'd'
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x_3 = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(x)
        x_3 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x_3)
        x_3 = Activation('relu')(x_3)

        x_3 = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x_3)
        x_3 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x_3)
        x_3 = Activation('relu')(x_3)

        x_3 = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x_3)
        x_3 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x_3)

        # block 25
        # block 25
        x = layers.add([x_3, x])
        x = Activation('relu')(x)

        model = Model(inputs=input, outputs=x)
        return model

    # (14, 14, 1024)
    def __func8__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        import keras.backend as K

        input = Input(shape=input_shape)
        # block 26 identify_block input_tensor, kernel_size, filters, stage, block
        input_tensor = input
        kernel_size = 3
        filters = [256, 256, 1024]
        stage = 4
        block = 'e'
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        model = Model(inputs=input, outputs=x)
        return model

    # [(14, 14, 1024), (14, 14, 1024)]
    def __func9__(self, input_shape):
        from keras.layers import Input
        from keras import layers
        from keras.layers import Activation
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        import keras.backend as K

        input = Input(shape=input_shape[0])
        input_tensor = Input(shape=input_shape[1])

        # block 27
        x = layers.add([input, input_tensor])
        x = Activation('relu')(x)

        # block 28 identify_block input_tensor, kernel_size, filters, stage, block
        kernel_size = 3
        filters = [256, 256, 1024]
        stage = 4
        block = 'f'
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x_1 = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(x)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x_1)
        x_1 = Activation('relu')(x_1)

        x_1 = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x_1)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x_1)
        x_1 = Activation('relu')(x_1)

        x_1 = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x_1)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x_1)

        # block 29
        x = layers.add([x_1, x])
        x = Activation('relu')(x)

        model = Model(inputs=[input, input_tensor], outputs=x)
        return model

    # (14, 14, 1024)
    def __func10__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        import keras.backend as K

        input = Input(shape=input_shape)
        # block 30 conv_block input_tensor, kernel_size, filters, stage, block, strides=(2, 2)
        input_tensor = input
        kernel_size = 3
        filters = [512, 512, 2048]
        stage = 5
        block = 'a'
        strides = (2, 2)
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        model = Model(inputs=input, outputs=x)
        return model

    # ((14, 14, 1024))
    def __func11__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        import keras.backend as K

        input_tensor = Input(shape=input_shape)
        kernel_size = 3
        filters = [512, 512, 2048]
        stage = 5
        block = 'a'
        strides = (2, 2)
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # block 31
        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        model = Model(inputs=input_tensor, outputs=shortcut)
        return model

    # [(7, 7, 2048), (7, 7, 2048)]
    def __func12__(self, input_shape):
        from keras.layers import Input
        from keras import layers
        from keras.layers import Activation
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        import keras.backend as K

        # block 32
        input = Input(shape=input_shape[0])
        shortcut = Input(shape=input_shape[1])

        x = layers.add([input, shortcut])
        x = Activation('relu')(x)

        # block 33
        kernel_size = 3
        filters = [512, 512, 2048]
        stage = 5
        block = 'b'
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x_1 = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(x)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x_1)
        x_1 = Activation('relu')(x_1)

        x_1 = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x_1)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x_1)
        x_1 = Activation('relu')(x_1)

        x_1 = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x_1)
        x_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x_1)

        # block 34
        # block 34
        x = layers.add([x, x_1])
        x = Activation('relu')(x)

        # block 35
        kernel_size = 3
        filters = [512, 512, 2048]
        stage = 5
        block = 'c'
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x_2 = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(x)
        x_2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x_2)
        x_2 = Activation('relu')(x_2)

        x_2 = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x_2)
        x_2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x_2)
        x_2 = Activation('relu')(x_2)

        x_2 = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x_2)
        x_2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x_2)

        # block 36
        # block 36
        x = layers.add([x_2, x])
        x = Activation('relu')(x)

        model = Model(inputs=[input, shortcut], outputs=x)
        return model

    # (7, 7, 2048)
    def __func13__(self, input_shape):
        from keras.layers import Input
        from keras.layers import Dense
        from keras.layers import Flatten
        from keras.layers import AveragePooling2D
        from keras.models import Model

        input = Input(shape=input_shape)
        # block 37
        x = AveragePooling2D((7, 7), name='avg_pool')(input)
        x = Flatten()(x)
        x = Dense(1000, activation='softmax', name='fc1000')(x)

        model = Model(inputs=input, outputs=x)
        return model
    def __init__(self):
        import numpy as np
        from Executer.resnet50 import resnet50
        from Executer.resnet50 import util_resnet50
        self.operations = []
        weights_model = resnet50(input_shape=(224, 224, 3)).model

        operation0 = operation(0, self.__func0__, (224, 224, 3), weights_model)
        operation1 = operation(1, self.__func1__, (55, 55, 64), weights_model)
        operation2 = operation(2, self.__func2__, (55, 55, 256), weights_model)
        operation3 = operation(3, self.__func3__, [(55, 55, 256), (55, 55, 256)], weights_model)
        operation4 = operation(4, self.__func4__, (55, 55, 256), weights_model)
        operation5 = operation(5, self.__func5__, (28, 28, 512), weights_model)
        operation6 = operation(6, self.__func6__, [(28, 28, 512), (28, 28, 512)], weights_model)
        operation7 = operation(7, self.__func7__, (28, 28, 512), weights_model)
        operation8 = operation(8, self.__func8__, (14, 14, 1024), weights_model)
        operation9 = operation(9, self.__func9__, [(14, 14, 1024), (14, 14, 1024)], weights_model)
        operation10 = operation(10, self.__func10__, (14, 14, 1024), weights_model)
        operation11 = operation(11, self.__func11__, (14, 14, 1024), weights_model)
        operation12 = operation(12, self.__func12__, [(7, 7, 2048), (7, 7, 2048)], weights_model)
        operation13 = operation(13, self.__func13__, (7, 7, 2048), weights_model)


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

    def excute(self, operationid, inputdata):
        return self.operations[operationid].excute(inputdata)
