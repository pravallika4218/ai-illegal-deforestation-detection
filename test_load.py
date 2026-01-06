import tensorflow as tf

model = tf.keras.models.load_model("best_deforestation_model.h5")
print("Model loaded OK")
