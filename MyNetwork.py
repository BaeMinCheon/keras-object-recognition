import keras

def Build(_wid, _hei, _dep, _cls, _act):

    input_shape = ()
    channel_dimension = 0

    if keras.backend.image_data_format() == "channels_first":
        input_shape = (_dep, _hei, _wid)
        channel_dimension = 1
    else:
        input_shape = (_hei, _wid, _dep)
        channel_dimension = -1

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization(axis=channel_dimension))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(128, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization(axis=channel_dimension))
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization(axis=channel_dimension))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(128, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization(axis=channel_dimension))
    model.add(keras.layers.Conv2D(128, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization(axis=channel_dimension))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(_cls))
    model.add(keras.layers.Activation(_act))

    return model