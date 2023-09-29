from collections import namedtuple
import altair as alt
import math
import glob
import pandas as pd
import streamlit as st
import openai
from qa import speechtotext
from mutagen.wave import WAVE
import os
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

import streamlit as st
from audiorecorder import audiorecorder

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

# Function for generating LLM response
def generate_response(speech_input, email, passwd):
     question0=speech_input
     question=speech_input
     query = speechtotext(speech_input)
     
     #ans, context, keys = chatbot_slim(query, text_split)
     return query
    
def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: display;}
      footer {visibility: display;}
     .stApp { bottom: 105px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 50, 0, 50),
        width=percent(100),
        color="black",
        text_align="left",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(1.5)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

 
from PIL import Image


import os
SECRET_TOKEN = os.environ["SECRET_TOKEN"] 
openai.api_key = SECRET_TOKEN


#-------------------------------------------------------------------------#
#--------------------------GUI CONFIGS------------------------------------#
#-------------------------------------------------------------------------#
# App title
st.set_page_config(page_title="🤗💬 SpeeKAR @ Gen AI-Chat Bot")
st.header("SpeeKAR @ Gen AI-Chat Bot")
st.title("Audio Recorder")
audio = audiorecorder("Click to record", "Click to stop recording")


# Hugging Face Credentials
with st.sidebar:
    st.title('🤗💬SpeeKAR @ Gen-AI Chat Bot')
    st.success('Access to this Gen-AI Powered Chatbot is provided by  [Anupam](https://www.linkedin.com/in/anupamisb/)!!', icon='✅')
    hf_email = 'anupam_purwar2019@pgp.isb.edu'
    hf_pass = 'PASS'
    st.markdown('📖 This app is hosted by Anupam Purwar [website](https://anupam-purwar.github.io/page/)!')
    image = Image.open('Diagram(1).jpg')
    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')


#------------------------------------------------------------------------------#
#-------------------------QUERY AUDIO INPUT------------------------------------#
#------------------------------------------------------------------------------#
if not audio.empty():
    # To play audio in frontend:
    st.audio(audio.export().read())  

    # To save audio to a file, use pydub export method:
    audio.export("query.wav", format="wav")

    # To get audio properties, use pydub AudioSegment properties:
    st.write(f"Duration: {audio.duration_seconds} seconds")
    
    #st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
    querywav = WAVE("query.wav")
    if querywav.info.length > 0:
        query = generate_response("query.wav",hf_email,hf_pass) 
        st.markdown("""
            <style>
            .big-font {
                font-size:30px !important;
            }
            </style>
            """, unsafe_allow_html=True)
        st.markdown("Your question in text ::")
        st.write(query)



# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask anything about uploaded document ..."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

from io import StringIO

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    #st.write(string_data)




# User-provided prompt
page_bg_img = '''
<style>
body {
background-image: url("https://csrbox.org/media/Hero-Image.png");
background-size: cover;
}
</style>
'''

#st.markdown(page_bg_img, unsafe_allow_html=True)

#if prompt := st.chat_input():
#    st.session_state.messages.append({"role": "user", "content": prompt})
#    with st.chat_message("user"):
#        st.write(prompt)

# Generate a new response if last message is not from assistant
#if st.session_state.messages[-1]["role"] != "assistant":
#    with st.chat_message("assistant"):
#        with st.spinner("Thinking..."):
#            query = generate_response("query.wav",hf_email,hf_pass) 
#            st.write(query) 
#    message = {"role": "assistant", "content": query}
#    st.session_state.messages.append(message)
#if not audio.empty():


myargs = [
    "Made in India",""
    " with ❤️ by ",
    link("https://www.linkedin.com/in/anupamisb/", "@Anupam"),
     br(),
     link("https://anupam-purwar.github.io/page/", "SpeeKAR ChatBoT"),
    ]

def footer():
    myargs = [
    "Made in India",""
    " with ❤️ by ",
    link("https://www.linkedin.com/in/anupamisb/", " Anupam for "),
    link("https://anupam-purwar.github.io/page/", "SpeeKAR ChatBoT"),
    ]
    layout(*myargs)
  
footer()
