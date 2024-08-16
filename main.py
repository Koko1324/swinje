import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import sounddevice as sd

# Streamlit 페이지 설정
st.set_page_config(layout="wide")

# 로고 이미지 로드 및 레이아웃 설정
logo = Image.open("logo.png")
header_col1, header_col2 = st.columns([1, 6])
with header_col1:
    st.image(logo, use_column_width=True)
with header_col2:
    option = st.selectbox(
        "공부 모드 선택",
        ["캠 공부", "데시벨 공부", "캠+데시벨 공부"]
    )

model = tf.keras.models.load_model('sleep_detection_model.h5')

# 웹캠 예측 기능
if option == "캠 공부" or option == "캠+데시벨 공부":
    st.write("웹캠을 통해 실시간 예측을 시작합니다.")
    
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
            elif a == 0:
                cv2.putText(frame, "Awake", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 프레임을 화면에 표시
        cv2.imshow('Sleep Detection', frame)

        # 'q' 키를 누르면 루프를 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 모든 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# 데시벨 측정 기능
if option == "데시벨 공부" or option == "캠+데시벨 공부":
    st.write("주변 소리의 데시벨을 측정합니다.")
    
    def calculate_decibel_level(audio_data):
        """ 오디오 데이터에서 데시벨 수준을 계산하는 함수 """
        rms = np.sqrt(np.mean(np.square(audio_data)))  # RMS 계산
        if rms > 0:
            db = 20 * np.log10(rms)  # 데시벨로 변환
        else:
            db = 0
        return db

    def audio_callback(indata, frames, time, status):
        """ 오디오 스트림에서 호출되는 콜백 함수 """
        if status:
            print(status, flush=True)
        audio_data = indata[:, 0]  # 모노 채널 선택
        db_level = calculate_decibel_level(audio_data)  # 데시벨 수준 계산
        print(f"Decibel Level: {db_level:.2f} dB")  # 콘솔에 출력
        global b
        b = 1 if db_level > 80 else 0  # 데시벨이 80을 초과하면 b에 1 저장
        print(f"Variable b: {b}")

    # 변수 초기화
    b = 0

    # 오디오 스트림 설정
    try:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100):
            print("Listening...")
            sd.sleep(10000)  # 10초 동안 오디오 입력 처리
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
    if b == 1:
        st.title("공부하기 적절하지 않은 데시벨의 소음이 존재하므로 공부 장소를 옮기시는 것을 추천드립니다.")
