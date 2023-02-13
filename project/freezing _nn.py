from keras import models
from keras import optimizers
from keras import layers
from keras import applications
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import datetime

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_dir="/home/arseniy/main/prepared_data/train"
val_dir="/home/arseniy/main/prepared_data/validation"

gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest'
)

gen_val = ImageDataGenerator(
    rescale=1./255
)

data_gen_train=gen_train.flow_from_directory(
    train_dir,
    target_size=(150,150),
    class_mode='binary',
    batch_size=5
)

data_gen_val=gen_val.flow_from_directory(
    val_dir,
    target_size=(150,150),
    class_mode='binary',
    batch_size=5
)

freez_model=applications.VGG16(weights='imagenet',include_top=False, input_shape=(150,150,3))
freez_model.trainable= False

model=models.Sequential()
model.add(freez_model)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.02)))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

log_dir = "logs/freez_model/" + datetime.datetime.now().strftime("%Y%m%d + reg l2 0.02")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit_generator(
    data_gen_train,
    steps_per_epoch=100,
    epochs=30,
    validation_data=data_gen_val,
    validation_steps=200,
    callbacks=[tensorboard_callback]
    )

model.save("/home/arseniy/main/testprojectgit/project/saved_model/fm.h5")