import os
import cv2
original_dataset_dir = '/home/arseniy/main/all_data/train'
base_dir = '/home/arseniy/main/prepared_data'
os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

train_size=3000
val_size=1000
test_size=1000


for i in range(train_size//2):
    src = os.path.join(original_dataset_dir, "cat.{}.jpg".format(i))
    dst = os.path.join(train_cats_dir, "{}.jpg".format(i))
    cv2.imwrite(dst, cv2.imread(src))

    src = os.path.join(original_dataset_dir, "dog.{}.jpg".format(i))
    dst = os.path.join(train_dogs_dir, "{}.jpg".format(i))
    cv2.imwrite(dst, cv2.imread(src))

for i in range(train_size//2 , train_size//2 + val_size//2):
    src = os.path.join(original_dataset_dir, "cat.{}.jpg".format(i))
    dst = os.path.join(validation_cats_dir, "{}.jpg".format(i - train_size//2))
    cv2.imwrite(dst, cv2.imread(src))

    src = os.path.join(original_dataset_dir, "dog.{}.jpg".format(i))
    dst = os.path.join(validation_dogs_dir, "{}.jpg".format(i - train_size//2))
    cv2.imwrite(dst, cv2.imread(src))

for i in range(train_size//2 + val_size//2, train_size//2 + val_size//2 + test_size//2):
    src = os.path.join(original_dataset_dir, "cat.{}.jpg".format(i))
    dst = os.path.join(test_dir, "{}.jpg".format(i - train_size//2 - val_size//2))
    cv2.imwrite(dst, cv2.imread(src))

    src = os.path.join(original_dataset_dir, "dog.{}.jpg".format(i))
    dst = os.path.join(test_dir, "{}.jpg".format(i - train_size//2 - val_size//2 + test_size//2))
    cv2.imwrite(dst, cv2.imread(src))