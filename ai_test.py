import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

model = tf.keras.models.load_model('sleep_detection_model.h5')

# 테스트 데이터 준비
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'C:/Users/user/Desktop/swinje/swinje_test',  # 테스트 데이터가 저장된 디렉토리 경로
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# 모델 평가
loss, accuracy = model.evaluate(test_generator)
st.write(f"Test Loss: {loss}")
st.write(f"Test Accuracy: {accuracy}")
