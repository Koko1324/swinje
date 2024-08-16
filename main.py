import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sounddevice as sd
import time

# Streamlit 페이지 설정
st.set_page_config(layout="wide")

# 로고 이미지와 공부 모드 선택
logo = Image.open("logo.png")
header_col1, header_col2 = st.columns([1, 6])
with header_col1:
    st.image(logo, use_column_width=True)
with header_col2:
    option = st.selectbox(
        "공부 모드 선택",
        ["캠 공부", "데시벨 공부", "캠+데시벨 공부"],
        key='selectbox_option'
    )

# 타이머와 상태 초기화
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = 0
if 'is_studying' not in st.session_state:
    st.session_state.is_studying = False
if 'sleep_time' not in st.session_state:
    st.session_state.sleep_time = 0  # 잠을 잔 시간
if 'last_face_detected_time' not in st.session_state:
    st.session_state.last_face_detected_time = None  # 마지막으로 얼굴이 인식된 시간
if 'is_sleeping' not in st.session_state:
    st.session_state.is_sleeping = False
if 'db_level' not in st.session_state:  # db_level 초기화
    st.session_state.db_level = 0


# 타이머 표시 함수
def display_timer():
    elapsed_time = st.session_state.elapsed_time + (time.time() - st.session_state.start_time)
    sleep_time = st.session_state.sleep_time
    total_study_time = elapsed_time - sleep_time
    return time.strftime('%H:%M:%S', time.gmtime(total_study_time)), time.strftime('%H:%M:%S', time.gmtime(sleep_time))

# 공부 시작하기 버튼 클릭 시
if st.button("공부 시작하기", key="start_button"):
    st.session_state.start_time = time.time()
    st.session_state.is_studying = True
    st.session_state.last_face_detected_time = time.time()
    st.write("공부를 시작합니다!")

# 공부 그만하기 버튼 클릭 시
if st.button("공부 그만하기", key="stop_button"):
    if st.session_state.is_studying:
        st.session_state.elapsed_time += time.time() - st.session_state.start_time
        st.session_state.is_studying = False

        # 총 공부 시간에서 잠을 잔 시간을 뺀다
        total_study_time = st.session_state.elapsed_time - st.session_state.sleep_time
        if total_study_time < 0:
            total_study_time = 0

        st.write("공부를 종료합니다!")
        st.write(f"총 공부 시간: {time.strftime('%H:%M:%S', time.gmtime(total_study_time))}")
        st.write(f"잠을 잔 시간: {time.strftime('%H:%M:%S', time.gmtime(st.session_state.sleep_time))}")

# 캠 공부 기능
if option == "캠 공부" or option == "캠+데시벨 공부":
    if st.session_state.is_studying:
        st.write("웹캠을 통해 실시간 예측을 시작합니다.")
        
        # 얼굴 검출기 로드
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 웹캠 열기
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("웹캠을 열 수 없습니다.")
        else:
            stframe = st.empty()  # Streamlit에서 이미지를 표시할 공간
            timer_placeholder = st.empty()  # 타이머를 표시할 공간

            # 스트림 프레임을 위해 FPS 설정
            cap.set(cv2.CAP_PROP_FPS, 24)

            while st.session_state.is_studying:
                ret, frame = cap.read()
                if not ret:
                    st.write("비디오 프레임을 읽을 수 없습니다.")
                    break

                # 얼굴 검출
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) > 0:
                    # 얼굴이 감지되면 잠에서 깨어 있는 것으로 간주
                    st.session_state.last_face_detected_time = time.time()
                    st.session_state.is_sleeping = False
                    cv2.putText(frame, "Awake", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    # 얼굴이 1분 이상 감지되지 않으면 잠을 자는 것으로 간주
                    time_since_last_face_detected = time.time() - st.session_state.last_face_detected_time
                    if time_since_last_face_detected >= 60:  # 1분 이상 얼굴이 감지되지 않은 경우
                        if not st.session_state.is_sleeping:
                            st.session_state.is_sleeping = True
                            st.session_state.sleep_time += time_since_last_face_detected

                        cv2.putText(frame, "Sleeping or Not Present", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Streamlit에서 프레임 표시
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

                # 타이머 실시간 업데이트
                total_study_time_str, sleep_time_str = display_timer()
                timer_placeholder.metric(label="총 공부 시간", value=total_study_time_str)
                timer_placeholder.metric(label="잠을 잔 시간", value=sleep_time_str)

            cap.release()
            cv2.destroyAllWindows()

# 데시벨 측정 기능
if option == "데시벨 공부" or option == "캠+데시벨 공부":
    if st.session_state.is_studying:
        st.write("주변 소리의 데시벨을 실시간으로 측정합니다.")

        def calculate_decibel_level(audio_data):
            """ 오디오 데이터에서 데시벨 수준을 계산하는 함수 """
            if len(audio_data) == 0:
                return 0
            rms = np.sqrt(np.mean(np.square(audio_data)))
            return 20 * np.log10(rms) if rms > 0 else 0

        def audio_callback(indata, frames, time, status):
            """ 오디오 스트림에서 호출되는 콜백 함수 """
            if status:
                st.warning(f"Audio stream status: {status}")
            audio_data = indata[:, 0]
            if len(audio_data) > 0:  # Ensure audio_data is not empty
                db_level = calculate_decibel_level(audio_data)
                st.session_state.db_level = db_level

        # 실시간으로 데시벨을 표시할 공간
        db_placeholder = st.empty()

        try:
            with sd.InputStream(callback=audio_callback, channels=1, samplerate=44100):
                st.write("Listening...")
                while st.session_state.is_studying:
                    db_placeholder.write(f"Decibel Level: {st.session_state.db_level:.2f} dB")
                    time.sleep(0.5)  # 0.5초마다 업데이트
        except KeyboardInterrupt:
            st.write("Interrupted by user.")
        except Exception as e:
            st.error(f"Error: {e}")

        if st.session_state.db_level > 80:
            st.title("공부하기 적절하지 않은 데시벨의 소음이 존재하므로 공부 장소를 옮기시는 것을 추천드립니다.")
