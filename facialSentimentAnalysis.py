#Title: FACIAL SENTIMENT ANALYSIS
#Developer: Vishwas Puri
#Purpose: A program that detects your facial sentiment/expression from multiple types of emotions categorized in a pre-trained data set.

#This program is made using python supported by streamlit.
import streamlit as st
import cv2
st.set_page_config(layout="wide")
col = st.empty()
#library to get facial characteristics to determine emotions
from deepface import DeepFace
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
st.write("Press start to turn on camera and start making your facial sentiments!")

def facialSentimentAnalysis():
    class OpenCVVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            # converting image to bgra
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            #detecting faces in the image
            faces = faceCascade.detectMultiScale(gray, 1.1, 4)
            #highlighting the face with a rectangle around it
            cv2.rectangle(img, pt1=(0, 0), pt2=(700, 80), color=(0, 0, 0), thickness=-1)
            try:
                #analyzes the emotion of the face
                #writes the emotion on the image
                predictions = DeepFace.analyze(img, actions=['emotion'])
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, predictions['dominant_emotion'], (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
            except:
                cv2.putText(img, "No Face Detected!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # setting up streamlit camera configuration
    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_html_attrs={
            "style": {"margin": "0 auto", "border": "5px yellow solid"},
            "controls": False,
            "autoPlay": True,
        },
    )

    # Info Block
    st.write("If camera doesn't turn on, please ensure that your camera permissions are on!")
    with st.expander("Steps to enable permission"):
        st.write("1. Click the lock button at the top left of the page")
        st.write("2. Slide the camera slider to on")
        st.write("3. Reload your page!")

    st.subheader("Possible Output Emotions:")
    st.image("testimonials.jpg")

if __name__ == "__main__":
    facialSentimentAnalysis()
