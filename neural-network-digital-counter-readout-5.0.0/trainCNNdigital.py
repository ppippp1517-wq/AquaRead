import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import History
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os # <-- เพิ่มการ import ที่นี่

loss_ges = np.array([])
val_loss_ges = np.array([])

#%matplotlib inline
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

Input_dir = r'D:\projectCPE\neural-network-digital-counter-readout-5.0.0\ziffer_sortiert_resize'


x_data = []
y_data = []
subdir = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "NaN"]

for aktsubdir in subdir:
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png']:
        files.extend(glob.glob(os.path.join(Input_dir, aktsubdir, ext)))
    print(f"Looking in {os.path.join(Input_dir, aktsubdir)} : พบ {len(files)} files")  # debug ดูจำนวนไฟล์
    if aktsubdir == "NaN":
        category = 10
    else:
        category = aktsubdir
    for aktfile in files:
        try:
            test_image = Image.open(aktfile)
            test_image = np.array(test_image, dtype="float32")
            x_data.append(test_image)
            y_data.append(np.array([category]))
        except Exception as e:
            print(f"Could not read file {aktfile}: {e}")



x_data = np.array(x_data)
y_data = np.array(y_data)

# --- เพิ่มการตรวจสอบว่ามีข้อมูลโหลดเข้ามาหรือไม่ ---
if x_data.shape[0] == 0:
    print("ไม่พบไฟล์รูปภาพในไดเรกทอรีที่ระบุ กรุณาตรวจสอบเส้นทาง (path) ของ Input_dir")
    exit() # ออกจากโปรแกรมถ้าไม่พบข้อมูล

y_data = to_categorical(y_data, 11)
print(f"Shape of x_data: {x_data.shape}")
print(f"Shape of y_data: {y_data.shape}")

x_data, y_data = shuffle(x_data, y_data)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

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

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95), metrics = ["accuracy"])

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
                             rotation_range=Rotation_Angle)

train_iterator = datagen.flow(X_train, y_train, batch_size=Batch_Size) # ใช้ X_train, y_train
validation_iterator = datagen.flow(X_test, y_test, batch_size=Batch_Size) # ใช้ X_test, y_test

history = model.fit(train_iterator, validation_data = validation_iterator, epochs = Epoch_Anz)

loss_ges = np.append(loss_ges, history.history['loss'])
val_loss_ges = np.append(val_loss_ges, history.history['val_loss'])

plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','eval'], loc='upper left')
plt.show()

# --- ส่วนที่แก้ไข 2: การทดสอบผลลัพธ์ ---
subdir = ["NaN", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
res = []

for aktsubdir in subdir:
    path_pattern = os.path.join(Input_dir, aktsubdir, '*.jpg') # <-- ใช้ os.path.join
    files = glob.glob(path_pattern)
    if aktsubdir == "NaN":
        zw1 = -1
    else:
        zw1 = int(aktsubdir)
    for aktfile in files:
        test_image = Image.open(aktfile)
        test_image = np.array(test_image, dtype="float32")
        img = np.reshape(test_image,[1,32,20,3])
        prediction = model.predict(img)
        classes = np.argmax(prediction, axis=1)
        classes = classes[0]
        if classes == 10:
            classes = -1
        zw2 = classes
        zw3 = zw2 - zw1
        res.append(np.array([zw1, zw2, zw3]))

res = np.asarray(res)

plt.plot(res[:,0])
plt.plot(res[:,1])
plt.title('Result')
plt.ylabel('Digital Value')
plt.xlabel('#Picture')
plt.legend(['real','model'], loc='upper left')
plt.show()

model.save("Train_CNN_Digital-Readout_Version_5.0.0.h5")

# --- ส่วนที่แก้ไข 3: การแสดงไฟล์ที่ทำนายผิด ---
only_deviation = True

subdir = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "NaN"]

for aktsubdir in subdir:
    path_pattern = os.path.join(Input_dir, aktsubdir, '*.jpg') # <-- ใช้ os.path.join
    files = glob.glob(path_pattern)
    expected_class = aktsubdir
    for aktfile in files:
        test_image = Image.open(aktfile)
        test_image = np.array(test_image, dtype="float32")
        img = np.reshape(test_image,[1,32,20,3])
        prediction = model.predict(img)
        classes = np.argmax(prediction, axis=1)
        classes = classes[0]
        if classes == 10:
            classes = "NaN"
        if only_deviation == True:
            if str(classes) != str(expected_class):
                print(aktfile + " " + aktsubdir +  " " + str(classes))
        else:
            print(aktfile + " " + aktsubdir +  " " + str(classes))
