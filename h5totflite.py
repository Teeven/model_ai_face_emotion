import tensorflow as tf

# Memuat model dari file H5
model = tf.keras.models.load_model('model.h5')

# Membuat converter untuk TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Opsional: Menambahkan kuantisasi
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Mengkonversi model
tflite_model = converter.convert()

# Menyimpan model TFLite ke file
with open('model_file.tflite', 'wb') as f:
    f.write(tflite_model)
