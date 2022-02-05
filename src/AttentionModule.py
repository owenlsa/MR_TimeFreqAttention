from keras.layers import (Activation, Add, AveragePooling2D,
                          Concatenate, Conv2D, Dense, GlobalAveragePooling2D,
                          GlobalMaxPooling2D, Permute, Reshape, multiply)


def AttentionBlock(input_feature, attention="FTA", ratio=8, prefix="FTA"):
    if attention == "FTA":
        refined_feature = _FtaBlock(input_feature, ratio, prefix)
    elif attention == "CBAM":
        refined_feature = _CbamBlock(input_feature, ratio, prefix)
    elif attention == "CAM,FAM":
        refined_feature = _CamFamBlock(input_feature, ratio, prefix)
    elif attention == "CAM,TAM":
        refined_feature = _CamTamBlock(input_feature, ratio, prefix)
    elif attention == "CAM,FAM,TAM":
        refined_feature = _CamFamTamBlock(input_feature, ratio, prefix)
    else:
        raise Exception("Wrong attention type.")
    return refined_feature


def _FtaBlock(input_feature, ratio=8, prefix="FTA"):
    channel_feature = _ChannelAttention(input_feature, ratio, prefix)
    # f t attention separately
    freq_feature = _FrequencyAttention(channel_feature, prefix)
    time_feature = _TimeAttention(channel_feature, prefix)
    # f t attention merge
    concat = Concatenate(axis=-1, name=prefix +
                         '-FT_concat')([freq_feature, time_feature])
    out_feature = Conv2D(filters=freq_feature._keras_shape[-1],
                         kernel_size=1,
                         padding='same',
                         kernel_initializer='he_normal',
                         bias_initializer='zeros',
                         activation='relu',
                         name=prefix+'-FT_conv')(concat)
    return out_feature


def _CbamBlock(cbam_feature, ratio=8, prefix="CBAM"):
    cbam_feature = _ChannelAttention(cbam_feature, ratio, prefix)
    cbam_feature = _SpatialAttention(cbam_feature, prefix)
    return cbam_feature


def _CamFamBlock(input_feature, ratio=8, prefix="FTA"):
    channel_feature = _ChannelAttention(input_feature, ratio, prefix)
    freq_feature = _FrequencyAttention(channel_feature, prefix)
    return freq_feature


def _CamTamBlock(input_feature, ratio=8, prefix="FTA"):
    channel_feature = _ChannelAttention(input_feature, ratio, prefix)
    time_feature = _TimeAttention(channel_feature, prefix)
    return time_feature


def _CamFamTamBlock(input_feature, ratio=8, prefix="FTA"):
    channel_feature = _ChannelAttention(input_feature, ratio, prefix)
    freq_feature = _FrequencyAttention(channel_feature, prefix)
    time_feature = _TimeAttention(freq_feature, prefix)
    return time_feature


def _ChannelAttention(input_feature, ratio=8, prefix="FTA"):
    channel_axis = -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros',
                             name=prefix+'-Chan_dense1')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros',
                             name=prefix+'-Chan_dense2')

    avg_pool = GlobalAveragePooling2D(
        name=prefix+'-Chan_avgpool1')(input_feature)
    avg_pool = Reshape((1, 1, channel), name=prefix+'-Chan_reshape1')(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D(name=prefix+'-Chan_avgpool2')(input_feature)
    max_pool = Reshape((1, 1, channel), name=prefix+'-Chan_reshape2')(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    channel_feature = Add(name=prefix+'-Chan_add')([avg_pool, max_pool])
    channel_feature = Activation(
        'sigmoid', name=prefix+'-Chan_sigmoid')(channel_feature)

    return multiply([input_feature, channel_feature], name=prefix+'-Chan_multiply')


def _FrequencyAttention(input_feature, prefix="FTA"):
    kernel_size = 3

    input_feature_freq = AveragePooling2D(pool_size=(
        1, input_feature._keras_shape[2]), strides=1, name=prefix+'-Freq_avgpool1')(input_feature)

    avg_pool = Permute((3, 2, 1), name=prefix +
                       '-Freq_permute1')(input_feature_freq)
    avg_pool = GlobalAveragePooling2D(name=prefix+'-Freq_avgpool2')(avg_pool)
    avg_pool = Reshape(
        (avg_pool._keras_shape[1], 1, 1), name=prefix+'-Freq_reshape1')(avg_pool)
    # avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(input_feature_freq)
    assert avg_pool._keras_shape[-1] == 1

    max_pool = Permute((3, 2, 1), name=prefix +
                       '-Freq_permute2')(input_feature_freq)
    max_pool = GlobalMaxPooling2D(name=prefix+'-Freq_avgpol3')(max_pool)
    max_pool = Reshape(
        (max_pool._keras_shape[1], 1, 1), name=prefix+'-Freq_reshape2')(max_pool)
    # max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(input_feature_freq)
    assert max_pool._keras_shape[-1] == 1

    concat = Concatenate(axis=3, name=prefix +
                         '-Freq_concat')([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2

    freq_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name=prefix+'-Freq_conv1')(concat)
    freq_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name=prefix+'-Freq_conv2')(freq_feature)
    freq_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name=prefix+'-Freq_conv3')(freq_feature)
    assert freq_feature._keras_shape[-1] == 1

    return multiply([input_feature, freq_feature], name=prefix+'-Freq_multiply')


def _TimeAttention(input_feature, prefix="FTA"):
    kernel_size = 3

    input_feature_time = AveragePooling2D(pool_size=(input_feature._keras_shape[1], 1),
                                          strides=1, name=prefix+'-Time_avgpool1')(input_feature)

    avg_pool = Permute((1, 3, 2), name=prefix +
                       '-Time_permute1')(input_feature_time)
    avg_pool = GlobalAveragePooling2D(name=prefix + '-Time_avgpool2')(avg_pool)
    avg_pool = Reshape(
        (1, avg_pool._keras_shape[1], 1), name=prefix + '-Time_reshape1')(avg_pool)
    # avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_feature_time)
    assert avg_pool._keras_shape[-1] == 1

    max_pool = Permute((1, 3, 2), name=prefix +
                       '-Time_permute2')(input_feature_time)
    max_pool = GlobalMaxPooling2D(name=prefix + '-Time_avgpool3')(max_pool)
    max_pool = Reshape(
        (1, max_pool._keras_shape[1], 1), name=prefix + '-Time_reshape2')(max_pool)
    # max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_feature_time)
    assert max_pool._keras_shape[-1] == 1

    concat = Concatenate(axis=3, name=prefix +
                         '-Time_concat')([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2

    time_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name=prefix + '-Time_conv1')(concat)
    time_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name=prefix + '-Time_conv2')(time_feature)
    time_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name=prefix + '-Time_conv3')(time_feature)
    assert time_feature._keras_shape[-1] == 1

    return multiply([input_feature, time_feature],
                    name=prefix + '-Time_multiply')


def _SpatialAttention(input_feature, prefix="CBAM"):
    kernel_size = 7
    cbam_feature = input_feature

    avg_pool = Permute((1, 3, 2), name=prefix +
                       '-Spat_permute1')(cbam_feature)
    avg_pool = GlobalAveragePooling2D(name=prefix + '-Spat_avgpool2')(avg_pool)
    avg_pool = Reshape(
        (1, avg_pool._keras_shape[1], 1), name=prefix + '-Spat_reshape1')(avg_pool)
    # avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1

    max_pool = Permute((1, 3, 2), name=prefix +
                       '-Spat_permute2')(cbam_feature)
    max_pool = GlobalMaxPooling2D(name=prefix + '-Spat_avgpool3')(max_pool)
    max_pool = Reshape(
        (1, max_pool._keras_shape[1], 1), name=prefix + '-Spat_reshape2')(max_pool)
    # max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1

    concat = Concatenate(axis=3, name=prefix +
                         '-Spat_concat')([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2

    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name=prefix+'-Spat_conv1')(concat)
    assert cbam_feature._keras_shape[-1] == 1

    return multiply([input_feature, cbam_feature])
