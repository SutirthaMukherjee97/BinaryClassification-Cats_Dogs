# The concept of flow from directory has been utilised
## Image DataGenrator

train_dir = '/content/train'
test_dir = '/content/test1'

train_datagen = ImageDataGenerator(
    rescale = 1./ 255.,horizontal_flip=True,vertical_flip=False,rotation_range=45,brightness_range=[0.2,1.6]
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size = 32,
    class_mode = 'categorical',
    target_size = (224,224)
);


valid_datagen = ImageDataGenerator(
    rescale = 1./ 255.
)

validation_generator = valid_datagen.flow_from_directory(
    test_dir,
    batch_size = 16,
    class_mode = 'categorical',
    target_size = (224,224)
);
