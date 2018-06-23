import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.applications.vgg19 import VGG19
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

num_classes = 7
width, height = 48, 48

data = pd.read_csv("./data/fer2013.csv")


def convert_fer2013(data):
    pixels = data['pixels'].tolist()

    faces = []
    for pixel_sq in pixels:
        face = [int(pixel) for pixel in pixel_sq.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        face = np.stack((face,)*3, -1)
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    return faces, emotions


def normalize(imgs):
    new_imgs = []
    for img in imgs:
        img = img / 255.0
        new_imgs.append(img)

    return new_imgs


X, y = convert_fer2013(data)
X = np.array(normalize(X))

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=41,
                                                    stratify=y)

model = VGG19(include_top=False,
              weights='imagenet',
              pooling='avg')

for layer in model.layers:
    layer.trainable = False

x = model.output

y = Dense(num_classes,
          activation='softmax')(x)

final_model = Model(input=model.input,
                    output=y)

opt = SGD(lr=0.0005, momentum=0.9, nesterov=True)

final_model.compile(optimizer=opt,
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])

final_model.fit(X_train, y_train,
                batch_size=32,
                epochs=10,
                validation_split=0.1,
                shuffle=True,
                verbose=2)

for layer in final_model.layers[4:]:
    layer.trainable = True

final_model.compile(optimizer=opt,
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.9,
                               patience=6,
                               verbose=1)
tensorboard = TensorBoard(log_dir="./logs")
checkpointer = ModelCheckpoint(filepath='./models/model_v2.h5',
                               monitor='val_loss',
                               verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

train_generator = ImageDataGenerator(featurewise_center=True,
                                     featurewise_std_normalization=True,
                                     rotation_range=30,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     horizontal_flip=True)
train_generator.fit(X_train, augment=True, seed=4)

test_generator = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True)
test_generator.fit(X_train, augment=True, seed=4)

final_model.fit_generator(train_generator.flow(X_train, y_train,
                                               batch_size=32),
                          steps_per_epoch=len(X_train)/32,
                          epochs=100,
                          validation_data=test_generator.flow(X_test, y_test,
                                                              batch_size=32),
                          callbacks=[lr_reducer, tensorboard, checkpointer, early_stopper],
                          verbose=2)

scores = final_model.evaluate(X_test, y_test,
                              batch_size=32)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))
