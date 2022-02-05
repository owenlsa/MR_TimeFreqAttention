import keras
from keras import Model
from keras.initializers import lecun_normal, Zeros, he_normal
from keras.layers import (Activation, Add, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D,
                          Flatten, Input, MaxPooling2D, Multiply)

from AttentionModule import AttentionBlock


def ScModel(attention="", multi_input=False, input_shape=(100, 100, 3), categories_num=11, seeds=100, prefix='SC'):
    dr = 0.5

    if multi_input:
        model_name = 'DSF_' + prefix + '_model'
        ori_img_input = Input(shape=input_shape, name='originalInput')
        filt_img_input = Input(shape=input_shape, name='filteredInput')
        concat = Concatenate(
            axis=-1, name='dualInput-Concatenate')([ori_img_input, filt_img_input])
        compress = Conv2D(3, (1, 1),
                          kernel_initializer=lecun_normal(seed=seeds),
                          bias_initializer='zeros',
                          padding="same",
                          name='dualInput-Conv0')(concat)
        net_input = compress
    else:
        model_name = prefix + '_model'
        img_input = Input(shape=input_shape, name='Input')
        net_input = img_input

    x = Conv2D(64, (3, 3),
               kernel_initializer=lecun_normal(seed=seeds),
               bias_initializer='zeros',
               name=prefix+'-Conv1')(net_input)
    if attention:
        x = AttentionBlock(
            x, attention=attention, prefix=prefix+'-Conv1-FTA')
    x = Activation('relu', name=prefix+'-Conv1-Activation')(x)
    x = MaxPooling2D(pool_size=(2, 2), name=prefix+'-Maxpool1')(x)

    x = Conv2D(32, (3, 3),
               kernel_initializer=lecun_normal(seed=seeds),
               bias_initializer='zeros',
               name=prefix+'-Conv2')(x)
    if attention:
        x = AttentionBlock(
            x, attention=attention, prefix=prefix+'-Conv2-FTA')
    x = Activation('relu', name=prefix+'-Conv2-Activation')(x)
    x = MaxPooling2D(pool_size=(2, 2), name=prefix+'-Maxpool2')(x)

    x = Conv2D(12, (3, 3),
               kernel_initializer=lecun_normal(seed=seeds),
               bias_initializer='zeros',
               name=prefix+'-Conv3')(x)
    if attention:
        x = AttentionBlock(
            x, attention=attention, prefix=prefix+'-Conv3-FTA')
    x = Activation('relu', name=prefix+'-Conv3-Activation')(x)
    x = MaxPooling2D(pool_size=(2, 2), name=prefix+'-Maxpool3')(x)

    x = Conv2D(8, (3, 3),
               kernel_initializer=lecun_normal(seed=seeds),
               bias_initializer='zeros',
               name=prefix+'-Conv4')(x)
    if attention:
        x = AttentionBlock(
            x, attention=attention, prefix=prefix+'-Conv4-FTA')
    x = Activation('relu', name=prefix+'-Conv4-Activation')(x)

    x = Flatten(name=prefix+'-Flatten')(x)
    x = Dense(128,
              kernel_initializer=lecun_normal(seed=seeds),
              bias_initializer='zeros',
              name=prefix+'-Dense1')(x)
    x = Activation('relu', name=prefix+'-Dense1-Activation')(x)
    x = Dropout(dr, name=prefix+'-Dropout')(x)
    x = Dense(categories_num, name=prefix+'-Dense2')(x)
    output = Activation('softmax', name=prefix+'-Dense2-Activation')(x)

    if multi_input:
        model = Model(inputs=[ori_img_input, filt_img_input],
                      outputs=output, name=model_name)
    else:
        model = Model(inputs=img_input, outputs=output, name=model_name)

    return model


