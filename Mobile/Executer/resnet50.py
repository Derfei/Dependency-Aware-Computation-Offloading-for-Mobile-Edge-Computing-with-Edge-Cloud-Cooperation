# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
class util_resnet50:

    @classmethod
    def load_weights(cls, resnet50_model, model):
        for layer in model.layers:
            try:
                if resnet50_model.get_layer(layer.name) != None:
                    layer.set_weights(resnet50_model.get_layer(name=layer.name).get_weights())
            except Exception as e:
                print("cannot find the layer {0} in the vgg model and exception is {1}".format(layer.name,
                                                                                               e))
                pass
        pass

class resnet50:

    @classmethod
    def identity_block(cls, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filterss of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
        from keras import layers
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        import keras.backend as K

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

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    @classmethod
    def conv_block(cls, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """conv_block is the block that has a conv layer at shortcut

        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filterss of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.

        Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        And the shortcut should have strides=(2,2) as well
        """
        from keras import layers
        from keras.layers import Activation
        from keras.layers import Conv2D
        from keras.layers import BatchNormalization
        import keras.backend as K

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

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x


    def __init__(self, input_shape):
        from keras.layers import Input
        from keras import layers
        from keras.layers import Dense
        from keras.layers import Activation
        from keras.layers import Flatten
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.layers import GlobalMaxPooling2D
        from keras.layers import ZeroPadding2D
        from keras.layers import AveragePooling2D
        from keras.layers import GlobalAveragePooling2D
        from keras.layers import BatchNormalization
        from keras.models import Model
        from keras.preprocessing import image
        import keras.backend as K
        from keras.utils import layer_utils
        from keras.utils.data_utils import get_file
        from keras.applications.imagenet_utils import decode_predictions
        from keras.applications.imagenet_utils import preprocess_input
        from keras.applications.imagenet_utils import _obtain_input_shape
        from keras.engine.topology import get_source_inputs
        K.set_image_data_format('channels_last')
        WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

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

        self.model = Model(inputs=img_input, outputs=x)

        'load the weight'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models',
                                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')

        self.model.load_weights(weights_path)


    def plot_model(self):
        from keras.utils import plot_model
        plot_model(model=self.model, to_file='modelresnet50.png', show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    import numpy as np
    from keras.layers import Input
    from keras import layers
    from keras.layers import Dense
    from keras.layers import Activation
    from keras.layers import Flatten
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import GlobalMaxPooling2D
    from keras.layers import ZeroPadding2D
    from keras.layers import AveragePooling2D
    from keras.layers import GlobalAveragePooling2D
    from keras.layers import BatchNormalization
    from keras.models import Model
    from keras.preprocessing import image
    import keras.backend as K
    from keras.utils import layer_utils
    from keras.utils.data_utils import get_file
    from keras.applications.imagenet_utils import decode_predictions
    from keras.applications.imagenet_utils import preprocess_input
    from keras.applications.imagenet_utils import _obtain_input_shape
    from keras.engine.topology import get_source_inputs
    from keras.utils import plot_model

    'define the block we need'
    # block 0
    img_input = Input(shape=(224, 224, 3))
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # block 1 conv_block input_tensor, kernel_size, filters, stage, block, strides=(2, 2)
    input_tensor = x
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

    # block 2
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    # block 3
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    # block 4 identify_block input_tensor, kernel_size, filters, stage, block
    input_tensor = x
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

    #block 5
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    #block 6 input_tensor, kernel_size, filters, stage, block
    input_tensor = x
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

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # block 7
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    # block 8 conv_block input_tensor, kernel_size, filters, stage, block, strides=(2, 2)
    input_tensor = x
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

    # block 9
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    # block 10
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    # block 11 identify_block input_tensor, kernel_size, filters, stage, block
    input_tensor = x
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

    # block 12
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    # block 13 identify_block input_tensor, kernel_size, filters, stage, block
    input_tensor = x
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

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # block 14
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    # block 15 identify_block input_tensor, kernel_size, filters, stage, block
    input_tensor = x
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

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # block 16
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    # block 17 conv_block input_tensor, kernel_size, filters, stage, block, strides=(2, 2)
    input_tensor = x
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
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    # block 19
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    # block 20 identify_block input_tensor, kernel_size, filters, stage, block
    input_tensor = x
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

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # block 21
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    # block 22 identify_block input_tensor, kernel_size, filters, stage, block
    input_tensor = x
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

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # block 23
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    # block 24 identify_block input_tensor, kernel_size, filters, stage, block
    input_tensor = x
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

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # block 25
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    # block 26 identify_block input_tensor, kernel_size, filters, stage, block
    input_tensor = x
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

    # block 27
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    # block 28 identify_block input_tensor, kernel_size, filters, stage, block
    input_tensor = x
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

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # block 29
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    # block 30 conv_block input_tensor, kernel_size, filters, stage, block, strides=(2, 2)
    input_tensor = x
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

    # block 31
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    # block 32
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    # block 33 identify_block input_tensor, kernel_size, filters, stage, block
    input_tensor = x
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

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # block 34
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    # block 35 identify_block input_tensor, kernel_size, filters, stage, block
    input_tensor = x
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

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # block 36
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    # block 37
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(1000, activation='softmax', name='fc1000')(x)

    model = Model(inputs=img_input, outputs=x)

    'get the resnet50 model'
    resnet50_model = resnet50(input_shape=(224, 224, 3))
    resnet50_model.plot_model()

    'load the weight of the block'
    util_resnet50.load_weights(resnet50_model.model, model)

    'try to prodict the weight'
    input_data = np.array([np.zeros(shape=(224, 224, 3))])
    model.predict(input_data)

    plot_model(model, "modelresnet50test.png", show_shapes=True, show_layer_names=True)

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
    preds_resnet50 = resnet50_model.model.predict(x)
    print('Predicted:', decode_predictions(preds_resnet50))

