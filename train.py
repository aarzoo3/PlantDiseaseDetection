import tensorflow as tf
import pandas as pd
import seaborn as sns
import json


class Cnn:
    def __init__(self,train_data, validation_data):
        self. train_data = train_data
        self.validation_data = validation_data

    def define_data_sets(self):
        training_set = tf.keras.utils.image_dataset_from_directory(
            self.train_data,
            labels='inferred',
            color_mode='rgb',
            batch_size=32,
            image_size=(128, 128),
            shuffle=True,
            seed=None,
            interpolation='bilinear',
            validation_split=None,
            label_mode='categorical'
        )

        validation_set = tf.keras.utils.image_dataset_from_directory(
            self.validation_data,
            labels='inferred',
            color_mode='rgb',
            batch_size=32,
            image_size=(128, 128),
            shuffle=True,
            seed=None,
            interpolation='bilinear',
            validation_split=None,
            label_mode='categorical'

        )
        return training_set, validation_set

    def train(self,training_set,validation_set):
        cnn = tf.keras.models.Sequential()
        cnn.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)))
        cnn.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
        cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
        cnn.add(tf.keras.layers.Conv2D(64, (3,3), padding='same',activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
        cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
        cnn.add(tf.keras.layers.Conv2D(128, (3,3), padding='same',activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
        cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
        cnn.add(tf.keras.layers.Conv2D(256, (3,3), padding='same',activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))
        cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
        cnn.add(tf.keras.layers.Conv2D(512, (3,3), padding='same',activation='relu'))
        cnn.add(tf.keras.layers.Conv2D(512, (3,3), activation='relu'))
        cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
        cnn.add(tf.keras.layers.Dropout(0.25))
        cnn.add(tf.keras.layers.Flatten())
        cnn.add(tf.keras.layers.Dense(units=1500,activation='relu'))

        cnn.add(tf.keras.layers.Dropout(0.4)) #To avoid overfitting
        cnn.add(tf.keras.layers.Dense(38, activation='softmax'))
        cnn.compile(optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
        print(cnn.summary())
        training_history = cnn.fit(x=training_set, validation_data=validation_set, epochs=10)
        print('training history is -->',training_history.history)
        with open('training_hist_10epoch.json', 'w') as file_obj:
            json.dump(training_history.history, file_obj)
        cnn.save('trained_plant_disease_model_10epoch.keras')

        return cnn
    def evaluate(self, cnn, training_set, validation_set):
        train_loss, train_acc = cnn.evaluate(training_set)
        print('Training accuracy:', train_acc)
        val_loss, val_acc = cnn.evaluate(validation_set)
        print('Validation accuracy:', val_acc)
        return 0

training_file = r"C:\Users\lenovo\Downloads\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
validation_file = r"C:\Users\lenovo\Downloads\archive\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid"

x = Cnn(training_file, validation_file)
train, val = x.define_data_sets()
cn_model = x.train(train,val)
evv = x.evaluate(cn_model,train,val)
print(cn_model)
print(evv)
