import graphs as graph
from keras import models
from keras import optimizers
from keras import layers
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

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
    horizontal_flip=True,
    fill_mode='nearest'
)
gen_val = ImageDataGenerator(
    rescale=1./255
)
data_gen_train=gen_train.flow_from_directory(
    train_dir,
    target_size=(150,150),
    class_mode='binary',
    batch_size=20
)
data_gen_val=gen_val.flow_from_directory(
    val_dir,
    target_size=(150,150),
    class_mode='binary',
    batch_size=20
)

freez_model=applications.VGG16(weights='imagenet',include_top=False, input_shape=(150,150,3))
freez_model.trainable= False

model=models.Sequential()
model.add(freez_model)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

history = model.fit_generator(
    data_gen_train,
    steps_per_epoch=100,
    epochs=30,
    validation_data=data_gen_val,
    validation_steps=50
    )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

graph.main(acc,loss,val_acc,val_loss)

model.save("/home/arseniy/main/testprojectgit/project/saved_model/fm.h5")