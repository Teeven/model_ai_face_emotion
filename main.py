# Mengimpor paket yang diperlukan
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

# Direktori data pelatihan dan validasi
train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'

# Augmentasi data untuk data pelatihan
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Menskalakan nilai piksel menjadi [0, 1]
    rotation_range=30,  # Memutar gambar secara acak hingga 30 derajat
    shear_range=0.3,  # Intensitas geser (sudut geser dalam radian berlawanan arah jarum jam)
    zoom_range=0.3,  # Memperbesar gambar secara acak
    horizontal_flip=True,  # Membalik gambar secara horizontal secara acak
    fill_mode='nearest'  # Mengisi piksel yang baru dibuat oleh transformasi di atas
)

# Augmentasi data untuk data validasi (hanya rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Generator untuk data pelatihan dan validasi
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',  # Mengubah gambar menjadi grayscale
    target_size=(48, 48),  # Mengubah ukuran gambar menjadi 48x48 piksel
    batch_size=32,
    class_mode='categorical',  # Mengembalikan label one-hot encoded 2D
    shuffle=True  # Mengacak data
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Label kelas untuk emosi
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Contoh batch data
img, label = train_generator.__next__()

# Arsitektur model
model = Sequential()

# Lapisan konvolusi pertama
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
# Lapisan konvolusi kedua
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Lapisan max pooling
model.add(Dropout(0.1))  # Lapisan dropout untuk mencegah overfitting

# Lapisan konvolusi ketiga
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# Lapisan konvolusi keempat
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# Meratakan output dan menambahkan lapisan fully connected
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

# Lapisan output dengan 7 neuron (satu untuk setiap kelas) dan aktivasi softmax
model.add(Dense(7, activation='softmax'))

# Mengompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Menghitung jumlah gambar pelatihan dan validasi
train_path = "data/train/"
test_path = "data/test"

num_train_imgs = sum([len(files) for r, d, files in os.walk(train_path)])
num_test_imgs = sum([len(files) for r, d, files in os.walk(test_path)])

print(num_train_imgs)
print(num_test_imgs)

# Melatih model
epochs = 30
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs//32,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_test_imgs//32
)

# Menyimpan model yang telah dilatih
model.save('model_file.h5')
