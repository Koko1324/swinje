import streamlit as st
from PIL import Image

#페이지 레이아웃 설정
st.set_page_config(layout="wide")

#로고 이미지를 열고 크기 조절하기
logo = Image.open("logo.png")

#레이아웃(헤더) 설정
header_col1, header_col2 = st.columns([1,6])

#벡엔드 프로그래밍-----------------------------------------------------------------------------------------
import tensorflow as tf  # TensorFlow 라이브러리 불러오기
from tensorflow.keras.models import Sequential  # Sequential 모델 불러오기
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # CNN 레이어들 불러오기
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array  # 이미지 전처리 라이브러리
import numpy as np  # NumPy 라이브러리
import cv2  # OpenCV 라이브러리 (카메라 영상 처리)

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

### 이미지 파일을 사용한 학습 단계 ###
# 학습 데이터 준비 (이미지 파일이 들어 있는 디렉토리 경로를 지정)
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'path_to_training_data',  # 학습 데이터가 저장된 디렉토리 경로
    target_size=(128, 128),  # 이미지 크기 조정
    batch_size=32,
    class_mode='binary'  # 이진 분류
)

# 모델 학습
model.fit(train_generator, epochs=10)

# 학습된 모델 저장
model.save('sleep_detection_model.h5')

### 웹캠 또는 스마트폰 카메라를 사용한 예측 단계 ###
# 학습된 모델 로드
model.load_weights('sleep_detection_model.h5')

# Haar Cascade 얼굴 검출기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 웹캠을 통해 실시간으로 영상을 가져옵니다.
cap = cv2.VideoCapture(0)  # 카메라를 열어 캡처 시작

while True:
    ret, frame = cap.read()  # 비디오 프레임을 읽어옵니다.
    if not ret:
        break

    # 프레임을 회색조로 변환하여 얼굴 검출
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        a = 2  # 얼굴이 인식되지 않으면 a에 2 저장
        cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    else:
        (x, y, w, h) = faces[0]  # 첫 번째 얼굴 영역 선택
        face_roi = frame[y:y+h, x:x+w]

        # 얼굴 영역을 128x128 크기로 조정
        resized_frame = cv2.resize(face_roi, (128, 128))
        img_array = img_to_array(resized_frame)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # 정규화

        prediction = model.predict(img_array)
        a = 1 if prediction[0] > 0.5 else 0  # 예측에 따라 a 설정

        # 결과를 프레임에 출력
        if a == 1:
            cv2.putText(frame, "Sleeping", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Awake", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 프레임을 화면에 표시
    cv2.imshow('Sleep Detection', frame)

    # 'q' 키를 누르면 루프를 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 자원 해제
cap.release()
cv2.destroyAllWindows()
