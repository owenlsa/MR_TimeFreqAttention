from keras.preprocessing.image import ImageDataGenerator


def DataGenerator(data_dir, target_size=(100, 100), batch_size=20, shuffle=True, seed=100):
    img_gen = ImageDataGenerator(rescale=1. / 255)
    data_generator = img_gen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle,
        seed=seed)
    samples_nums = len(data_generator.classes)

    return data_generator, samples_nums

