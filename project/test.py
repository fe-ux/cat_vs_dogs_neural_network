import tensorflow as tf
import cv2
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

new_model = tf.keras.models.load_model('/home/arseniy/main/testprojectgit/project/saved_model/model.h5')
size_test=1000
acc=0

t=np.zeros((250,150,150,3))
for j in range(size_test//250):
    for i in range(250):
        m=cv2.imread('/home/arseniy/main/prepared_data/test/{0}.jpg'.format(i+j*250))
        t[i]=cv2.resize(m,(150,150), interpolation = cv2.INTER_AREA)
    acc+=tf.math.reduce_sum(new_model.predict(t))

print(acc/size_test)


