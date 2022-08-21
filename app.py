import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional
import os
import pandas as pd
from datetime import  datetime
import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)

##
import face_recognition as face_rec
import cv2
import shutil
path = 'employee images'
employeeImg = []

employeeName = []
myList = os.listdir(path)
filename = 'click'


def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)



def findEncoding(images) :
    imgEncodings = []
    for img in images :
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings
def MarkAttendence(name):
    with open('attendence.csv', 'r+') as f:
        myDatalist =  f.readlines()
        nameList = []
        for line in myDatalist :
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.writelines(f'\n{name}, {timestr}')
            statment = str('welcome to seasia' + name)

for cl in myList :
    curimg = cv2.imread(f'{path}/{cl}')
    employeeImg.append(curimg)
    employeeName.append(os.path.splitext(cl)[0])

EncodeList = findEncoding(employeeImg)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)




HERE = Path(__file__).parent

logger = logging.getLogger(__name__)




def main():
    st.header("Attendence")

    pages = {

        "Attendence ": employ_recog,        
        "Admin ": Admin,       
    }
    page_titles = pages.keys()

    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    st.subheader(page_title)

    page_func = pages[page_title]
    page_func()

    st.sidebar.markdown(
        """
---
    """,  # noqa: E501
        unsafe_allow_html=True,
    )

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")





def emprec(img):    
    
    facesInFrame = face_rec.face_locations(img)

    encodeFacesInFrame = face_rec.face_encodings(img, facesInFrame)


    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        print(facedis)
        if min(facedis) < 0.5:
            matchIndex = np.argmin(facedis)

            print(matchIndex)


            name = employeeName[matchIndex].upper()
#             y1, x2, y2, x1 = faceloc
#             y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            top, right, bottom, left = faceloc
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name,  (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            print(name)        
            MarkAttendence(name)
            
    return img   

        
def Admin():
    result = 0
    
    user = st.sidebar.text_input('Username')
    passwd = st.sidebar.text_input('Password',type='password')
    if st.sidebar.checkbox('Login') :

        if user == 'Beastinblack' and passwd == 'AkashRaj' :
            
            st.success("Logged In as {}".format(user))
            result = 1

            # Tasks For Only Logged In Users
            if result == 1:
                if st.button('show attendence'):
                    st.dataframe(pd.read_csv('attendence.csv'))
                if st.button('clear attendence'):
                    dx = pd.DataFrame() 
                    dx.to_csv('attendence.csv') 
                if st.button('Add User'):
                    employee_name = st.sidebar.text_input('Employee Name')
                    img_file_buffer = st.camera_input("Register")
                    if img_file_buffer is not None:
                        bytes_data = img_file_buffer.getvalue()
                        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)  
                        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                        img = np.array(cv2_img)
                        st.image(img)
                        if st.button('Register Employee'):
                            cv2.imwrite(f'employee images/{employee_name}.jpg', img)
                            st.success("{employee_name} Registered")
           
    
    
    




    
def employ_recog():


    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = emprec(img)


        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="attendence",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )









if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
