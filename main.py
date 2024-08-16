import streamlit as st

#페이지 레이아웃 설정
st.set_page_config(layout="wide")

#인사말
header_col1, header_col2, header_col3 = st.columns([1,1,1])
with header_col2:
    st.write("공부 시간 측정 도우미")
