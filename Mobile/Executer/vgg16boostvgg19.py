# -*- coding: utf-8 -*-
'''
@author: longxin
@version: 1.0
@date:
@changeVersion:
@changeAuthor:
@description:
'''
class util_vgg16boostvgg19:

    @classmethod
    def get_vgg16model(cls):
        WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
        from keras.models import Model
        from keras.layers import Flatten
        from keras.layers import Dense
        from keras.layers import Input
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.utils.data_utils import get_file

        WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

        img_input = Input(shape=(224, 224, 3))

        # Block 1

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

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='16_block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='16_block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='16_block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='16_block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='16_block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='16_block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='16_block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='16_block5_pool')(x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='16_fc1')(x)
        x = Dense(4096, activation='relu', name='16_fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

        model = Model(inputs=img_input, output=x, name='vgg16')

        'load model weights'
        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models')
        model.load_weights(weights_path)
        return  model

    @classmethod
    def get_vgg19model(cls):
        from keras.models import Model
        from keras.layers import Flatten
        from keras.layers import Dense
        from keras.layers import Input
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.utils.data_utils import get_file
        from keras import layers

        WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
        img_input = Input(shape=(224, 224, 3))
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

        # Block 4
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block4_conv1')(x_vgg19)
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block4_conv2')(x_vgg19)
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block4_conv3')(x_vgg19)
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block4_conv4')(x_vgg19)
        x_vgg19 = MaxPooling2D((2, 2), strides=(2, 2), name='19_block4_pool')(x_vgg19)

        # Block 5
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block5_conv1')(x_vgg19)
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block5_conv2')(x_vgg19)
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block5_conv3')(x_vgg19)
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block5_conv4')(x_vgg19)
        x_vgg19 = MaxPooling2D((2, 2), strides=(2, 2), name='19_block5_pool')(x_vgg19)

        # classification block
        x_vgg19 = Flatten(name='flatten')(x_vgg19)
        x_vgg19 = Dense(4096, activation='relu', name='19_fc1')(x_vgg19)
        x_vgg19 = Dense(4096, activation='relu', name='19_fc2')(x_vgg19)
        x_vgg19 = Dense(1000, activation='softmax', name='19_predictions')(x_vgg19)

        model = Model(inputs=img_input, outputs=x_vgg19)
        weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models')
        model.load_weights(weights_path)

        return model

    @classmethod
    def util_load_vgg16boostvgg19_weight(cls, vgg16_model, vgg19_model, model):
        for layer in model.layers:
            try:
                if vgg16_model.get_layer(layer.name) != None:
                    layer.set_weights(vgg16_model.get_layer(layer.name).get_weights())
                    continue
                if vgg19_model.get_layer(layer.name) != None:
                    layer.set_weights(vgg19_model.get_layer(layer.name).get_weights())
                    continue

            except Exception as e:
                print("not find the weight of layer {0} and error is {1}".format(layer.name, e))
                pass
        pass

class vgg16boostvgg19:

    def __init__(self):
        from keras.models import Model
        from keras.layers import Flatten
        from keras.layers import Dense
        from keras.layers import Input
        from keras.layers import Conv2D
        from keras.layers import MaxPooling2D
        from keras.utils.data_utils import get_file
        from keras import layers

        WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

        img_input = Input(shape=(224, 224, 3))

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

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='16_block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='16_block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='16_block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='16_block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='16_block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='16_block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='16_block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='16_block5_pool')(x)

        # Classification block
        x = Flatten(name='16_flatten')(x)
        x = Dense(4096, activation='relu', name='16_fc1')(x)
        x = Dense(4096, activation='relu', name='16_fc2')(x)
        x = Dense(1000, activation='softmax', name='16_predictions')(x)


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

        # Block 4
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block4_conv1')(x_vgg19)
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block4_conv2')(x_vgg19)
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block4_conv3')(x_vgg19)
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block4_conv4')(x_vgg19)
        x_vgg19 = MaxPooling2D((2, 2), strides=(2, 2), name='19_block4_pool')(x_vgg19)

        # Block 5
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block5_conv1')(x_vgg19)
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block5_conv2')(x_vgg19)
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block5_conv3')(x_vgg19)
        x_vgg19 = Conv2D(512, (3, 3), activation='relu', padding='same', name='19_block5_conv4')(x_vgg19)
        x_vgg19 = MaxPooling2D((2, 2), strides=(2, 2), name='19_block5_pool')(x_vgg19)

        #classification block
        x_vgg19 = Flatten(name='19_flatten')(x_vgg19)
        x_vgg19 = Dense(4096, activation='relu', name='19_fc1')(x_vgg19)
        x_vgg19 = Dense(4096, activation='relu', name='19_fc2')(x_vgg19)
        x_vgg19 = Dense(1000, activation='softmax', name='19_predictions')(x_vgg19)

        output = layers.add([x_vgg19, x])
        output = Dense(1000, activation='softmax', name='predictions')(output)

        self.model = Model(inputs=img_input, outputs=output)



    def plot_model(self):
        from keras.utils import  plot_model
        plot_model(self.model, "vgg16boostvgg19model.png", show_layer_names=True, show_shapes=True)
        pass


if __name__ == "__main__":
    from keras.preprocessing import image
    import numpy as np
    from keras.models import Model
    from keras.layers import Flatten
    from keras.layers import Dense
    from keras.layers import Input
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.utils.data_utils import get_file
    from keras.preprocessing import image
    from keras.applications.imagenet_utils import decode_predictions
    from keras.applications.imagenet_utils import preprocess_input
    vgg19_model = util_vgg16boostvgg19.get_vgg19model()
    vgg16_model = util_vgg16boostvgg19.get_vgg16model()

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = vgg16_model.predict(x)
    print('Predicted:', decode_predictions(preds))

    tmp = vgg16boostvgg19()
    model = tmp.model
    tmp.plot_model()

    util_vgg16boostvgg19.util_load_vgg16boostvgg19_weight(vgg16_model, vgg19_model, model)

    preds_model = model.predict(x)
    print('Predicted:', decode_predictions(preds))
