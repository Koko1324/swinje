import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# CNN 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # 이진 분류를 위한 출력층
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 학습 데이터 준비
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'C:/Users/user/Desktop/swinje/swinje_train',  # 학습 데이터가 저장된 디렉토리 경로
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# 모델 학습
history = model.fit(train_generator, epochs=10)  # 에포크 수를 10으로 줄임 (자원과 시간 절약을 위해)

# 정확도와 손실 시각화
st.write("모델 학습 과정 시각화")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 정확도 플롯
ax[0].plot(history.history['accuracy'])
ax[0].set_title('Model Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')

# 손실 플롯
ax[1].plot(history.history['loss'])
ax[1].set_title('Model Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')

st.pyplot(fig)

# 학습된 모델 저장
model.save('sleep_detection_model.h5')