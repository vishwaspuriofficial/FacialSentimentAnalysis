import streamlit as st
import mediapipe as mp
import cv2
from deepface import DeepFace
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# st.set_page_config(layout="wide")

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
import av

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
st.write("Press start to turn on Camera!")
st.write("If camera doesn't turn on, click the select device button, change the camera input and reload your screen!")

def handDetector():
    class OpenCVVideoProcessor(VideoProcessorBase):


        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            faces = faceCascade.detectMultiScale(gray, 1.1, 4)
            cv2.rectangle(img, pt1=(0, 0), pt2=(700, 80), color=(0, 0, 0), thickness=-1)
            try:
                predictions = DeepFace.analyze(img, actions=['emotion'])
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, predictions['dominant_emotion'], (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
            except:
                cv2.putText(img, "No Face Detected!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
            return av.VideoFrame.from_ndarray(img, format="bgr24")


    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_html_attrs={
            "style": {"margin": "0 100", "border": "5px yellow solid"},
            "controls": False,
            "autoPlay": True,
        },
    )

if __name__ == "__main__":
    handDetector()