from keras import models
import tensorflow as tf
from keras import applications
from keras import layers
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import datetime

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

new_model = tf.keras.models.load_model('/home/arseniy/main/cat_vs_dogs_neural_network/project/saved_model/fm.h5')

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
    batch_size=15
)
data_gen_val=gen_val.flow_from_directory(
    val_dir,
    target_size=(150,150),
    class_mode='binary',
    batch_size=10
)

freez_model=applications.VGG16(weights='imagenet',include_top=False, input_shape=(150,150,3))
freez_model.trainable= True
set_trainable = False
for layer in freez_model.layers:
    if layer.name[0:6] == 'block5':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model=models.Sequential()
model.add(freez_model)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.set_weights(new_model.get_weights())

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-5),
    metrics=['acc']
    )

log_dir = "logs/finall_model/" + datetime.datetime.now().strftime("%Y%m%d + reg l2 0.01 + batch_size 15 / -l2")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit_generator(
    data_gen_train,
    steps_per_epoch=100,
    epochs=30,
    validation_data=data_gen_val,
    validation_steps=50,
    callbacks=[tensorboard_callback]
    )

model.save("/home/arseniy/main/testprojectgit/project/saved_model/model.h5")