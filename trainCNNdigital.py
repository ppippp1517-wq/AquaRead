import os
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import Image

# ==== CONFIG PATH ====
MODEL_SAVE_PATH = r"D:\projectCPE\Train_CNN_Digital-Readout_Version_5.0.0.h5"
Input_dir = r"D:\projectCPE\neural-network-digital-counter-readout-5.0.0\ziffer_sortiert_resize"

print("Current working directory:", os.getcwd())
print("Model will be saved at:", MODEL_SAVE_PATH)

loss_ges = np.array([])
val_loss_ges = np.array([])

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

subdir = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "NaN"]
files = glob.glob(Input_dir + '/*.*')
x_data = []
y_data = []

# ==== LOAD DATA ====
for aktsubdir in subdir:
    files = glob.glob(Input_dir + '/' + aktsubdir + '/*.jpg')
    if aktsubdir == "NaN":
        category = 10
    else:
        category = int(aktsubdir)
    for aktfile in files:
        test_image = Image.open(aktfile).convert('RGB')
        test_image = test_image.resize((32,20))        # resize ให้แน่ใจว่าตรง
        test_image = np.array(test_image, dtype="float32")
        x_data.append(test_image)
        y_data.append(np.array([category]))

x_data = np.array(x_data)
y_data = np.array(y_data)
y_data = to_categorical(y_data, 11)
print(x_data.shape)
print(y_data.shape)

x_data, y_data = shuffle(x_data, y_data, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

# ==== MODEL ====
model = Sequential()
model.add(BatchNormalization(input_shape=(32,20,3)))
model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dense(11, activation = "softmax"))
model.summary()

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95),
    metrics=["accuracy"]
)

Batch_Size = 4
Epoch_Anz = 80
Shift_Range = 1
Brightness_Range = 0.3
Rotation_Angle = 10
ZoomRange = 0.4

datagen = ImageDataGenerator(width_shift_range=[-Shift_Range,Shift_Range], 
                             height_shift_range=[-Shift_Range,Shift_Range],
                             brightness_range=[1-Brightness_Range,1+Brightness_Range],
                             zoom_range=[1-ZoomRange, 1+ZoomRange],
                             rotation_range=Rotation_Angle,
                             validation_split=0.2)

train_iterator = datagen.flow(X_train, y_train, batch_size=Batch_Size)
validation_iterator = datagen.flow(X_test, y_test, batch_size=Batch_Size)

history = model.fit(train_iterator, validation_data=validation_iterator, epochs=Epoch_Anz)

loss_ges = np.append(loss_ges, history.history['loss'])
val_loss_ges = np.append(val_loss_ges, history.history['val_loss'])
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','eval'], loc='upper left')
plt.show()

# ==== SAVE MODEL ====
print("Saving model to:", MODEL_SAVE_PATH)
model.save(MODEL_SAVE_PATH)

# ==== CHECK FILE ====
if os.path.exists(MODEL_SAVE_PATH):
    print("Model saved at:", MODEL_SAVE_PATH)
else:
    print(" ERROR: Model file NOT FOUND at", MODEL_SAVE_PATH)
    print("โปรดเช็ค permission, path, และลองเปลี่ยนชื่อ path หรือ run เป็น admin")

# ==== EVALUATE MODEL (PLOT REAL vs. MODEL) ====
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("\n=== Evaluating Model On All Data ===")


Input_dir = r"D:\projectCPE\neural-network-digital-counter-readout-5.0.0\ziffer_sortiert_resize"
subdir = ["NaN", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
res = []
for aktsubdir in subdir:
    files = glob.glob(Input_dir + '/' + aktsubdir + '\*.jpg')
    if aktsubdir == "NaN":
        zw1 = -1
    else:
        zw1 = int(aktsubdir)
    for aktfile in files:
        test_image = Image.open(aktfile)
        test_image = np.array(test_image, dtype="float32")
        img = np.reshape(test_image,[1,32,20,3])
        pred = model.predict(img)
        classes = np.argmax(pred, axis=1)[0]
        if classes == 10: 
            classes = -1
        zw2 = classes
        zw3 = zw2 - zw1
        res.append(np.array([zw1, zw2, zw3]))
res = np.asarray(res)
plt.plot(res[:,0], label="real")
plt.plot(res[:,1], label="model")
plt.title('Result')
plt.ylabel('Digital Value')
plt.xlabel('#Picture')
plt.legend(['real','model'], loc='upper left')
plt.show()

Input_dir = r"D:\projectCPE\neural-network-digital-counter-readout-5.0.0\ziffer_sortiert_resize"
only_deviation = True

subdir = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "NaN"]

for aktsubdir in subdir:
    files = glob.glob(Input_dir + '/' + aktsubdir + '\*.jpg')
    expected_class = aktsubdir
    for aktfile in files:
        test_image = Image.open(aktfile)
        test_image = np.array(test_image, dtype="float32")
        img = np.reshape(test_image,[1,32,20,3])
        pred = model.predict(img)          # ได้ค่า shape [1,11]
        classes = np.argmax(pred, axis=1)[0]   # ดึง index class ที่ทำนายได้

        if classes == 10: 
            classes = "NaN"
        if only_deviation == True:
            if str(classes) != str(expected_class):
                print(aktfile + " " + aktsubdir +  " " + str(classes))
        else:
            print(aktfile + " " + aktsubdir +  " " + str(classes))
