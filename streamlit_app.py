from collections import namedtuple
import altair as alt

import os, time
import pandas as pd
import math
import glob
import base64
from io import StringIO

import openai
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
# -------------IMPORTING CORE FUNCTIONALITIES OF THE SpeeKAR_BOT-------------
from qa import (
    speechtotext,
    readdoc_splittext,
    readdoc_splittext_pdf,
    create_context,
    create_db,
    chatbot_slim,
    chatbot,
    texttospeech_raw,
)

# -------------------AUDIO FUNCTIONALITY-------------------------
from mutagen.wave import WAVE

# --------------------HTML BUILDER AND FUNCTIONALITIES-----------------------------------#
from htbuilder import (
    HtmlElement,
    div,
    ul,
    li,
    br,
    hr,
    a,
    p,
    img,
    styles,
    classes,
    fonts,
)
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

import streamlit as st
from audiorecorder import audiorecorder


from PIL import Image


# ------------------DEFAULTS--------------------#
SECRET_TOKEN = os.environ["SECRET_TOKEN"]
openai.api_key = SECRET_TOKEN


# -----------------------HELPER FUNCTIONS--------------------------#
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


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
        opacity=1,
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(1.5),
    )

    body = p()
    foot = div(style=style_div)(hr(style=style_hr), body)

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


# -------------------------------FUNCTIONS FOR KAR BASED RESPONSE GENERATION-------------#
def process_query(speech_input, email, passwd):
    question0 = speech_input
    question = speech_input
    query = speechtotext(speech_input)

    return query


def generate_kARanswer(query, text_split):
    ans, context, keys = chatbot_slim(query, text_split)
    return ans, context, keys


# -------------------------------------------------------------------------#
# --------------------------GUI CONFIGS------------------------------------#
# -------------------------------------------------------------------------#
# App title
st.set_page_config(page_title="🤗💬 SpeeKAR @ Gen AI-Chat Bot")
st.header("SpeeKAR @ Gen AI-Chat Bot")


# Hugging Face Credentials
with st.sidebar:
    st.title("🤗💬SpeeKAR @ Gen-AI Chat Bot")
    st.success(
        "Access to this Gen-AI Powered Chatbot is provided by  [Anupam](https://www.linkedin.com/in/anupamisb/)!!",
        icon="✅",
    )
    hf_email = "anupam_purwar2019@pgp.isb.edu"
    hf_pass = "PASS"
    st.markdown(
        "📖 This app is hosted by [Anupam Purwar](https://anupam-purwar.github.io/page/) and supports only pdf/doc files."
    )
    image = Image.open("speekar_logo.png")
    st.image(
        image,
        caption=None,
        width=None,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )


# ---------------------------------------------------------#
# -----------------UPLOAD THE SRC DOCUMENT-----------------#
# ---------------------------------------------------------#
st.title("Please let me know what you want to talk about by choosing a file below!")
if "uploaded_status" not in st.session_state:
    st.session_state["uploaded_status"] = False

uploaded_file = st.file_uploader(label = "")
if st.session_state["uploaded_status"] == False and uploaded_file is not None:
    create_db.clear()
    readdoc_splittext.clear()
    readdoc_splittext_pdf.clear()

if "query_counter" not in st.session_state:
    st.session_state["query_counter"] = 0
if "query_status" not in st.session_state:
    st.session_state["query_status"] = False
if "audio_input_status" not in st.session_state:
    st.session_state["audio_input_status"] = False
if "text_input_status" not in st.session_state:
    st.session_state["text_input_status"] = False

if "db_created" not in st.session_state:
    st.session_state["db_created"] = False
    
if (uploaded_file is not None):
    st.session_state["uploaded_status"] = True
elif uploaded_file is None:
    st.session_state["uploaded_status"] = False
    st.session_state["query_counter"] = 0
    st.session_state["db_created"] = False
    st.session_state["text_input_status"] = False
    st.session_state["query_status"] = False
    st.session_state["audio_input_status"] = False
    st.write("Dear user, clearing unnecesary data fo you to start afresh!!")
    create_db.clear()
    readdoc_splittext.clear()
    readdoc_splittext_pdf.clear()
    st.write("You can upload your document now!")


if "messages" not in st.session_state.keys():
    st.session_state.messages = []

if (uploaded_file is not None):
    file_path = os.path.join( os.getcwd(), uploaded_file.name)
    with open(file_path,"wb") as f: 
        f.write(uploaded_file.getbuffer())         
    st.success("Saved File")

    # print(file_path)
    filename = file_path

    if ".docx" in filename: #uploaded_file.name:
        all_text, text_split, text_chunk, headings, para_texts = readdoc_splittext(filename)#uploaded_file.name)
    elif (".doc" in filename) and (".docx" not in filename): #uploaded_file.name:
        all_text, text_split, text_chunk, headings, para_texts = readdoc_splittext(filename)#uploaded_file.name)
    elif ".pdf" in filename: #uploaded_file.name:
        all_text, text_split, text_chunk, headings, para_texts = readdoc_splittext_pdf(filename)#uploaded_file.name)
    
    with st.chat_message("assistant"):
        st.write("Hi! Getting your contexts ready for query! Please wait!")

    hf, db = create_db(text_chunk, uploaded_file.name)    
    
    st.session_state["db_created"] = True    

    if uploaded_file is not None and st.session_state["db_created"] == True:
        st.title("Ask me anything about the document!")

    
    # User-provided prompt
    #if prompt := st.chat_input():
    
    
    with st.chat_message("user"):
        #query_audio_placeholder = st.empty()
        #audio = audiorecorder("Click to record", "Click to stop recording")
        #query_placeholder = st.empty()
        query_text = st.text_area(label = "Let me know what you have in mind!")
    st.session_state.messages.append({"role": "user", "content": query_text})
    if query_text != "":# or not audio.empty() and not os.path.exists("query.wav"):
        if query_text != "":
            st.session_state["query_status"] = True
            st.session_state["text_input_status"] = True
            st.session_state["query_counter"] += 1
            
            
            query = query_text
            
            context, keywords = create_context(query, text_split, headings, para_texts)
            
            # Generate a new response if last message is not from assistant
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if len(context) < 2000:
                        ans, context, keys = chatbot_slim(query, context, keywords)
                        
                        if (ans=='I don\'t know.' or ans=='I don\'t know' ):
                            ans = chatbot(query,db)
                            message = {"role": "assistant", "content": ans}
                        else:
                            
                            message = {"role": "assistant", "content": ans}
                    else:
                        ans = chatbot(query,db)
                        
                        message = {"role": "assistant", "content": ans}
                        
                
                
                # -----------text to speech--------------------------#
                texttospeech_raw(ans, language="en")
                mymidia_placeholder = st.empty()
                with open("answer.wav", "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    b64 = base64.b64encode(audio_bytes).decode()
                    md = f"""
                         <audio controls autoplay="true">
                         <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                         </audio>
                         """
                    mymidia_placeholder.empty()
                    time.sleep(1)
                    mymidia_placeholder.markdown(md, unsafe_allow_html=True)
            st.session_state.messages.append(message)    
            st.session_state["query_status"] = False
            st.session_state["text_input_status"] = False
            st.session_state["audio_input_status"] = False

# ------------------------------------------------------------------------------#
# -------------------------QUERY AUDIO INPUT - RETURNING TEXT QUERY-------------#
# ------------------------------------------------------------------------------#

if st.session_state.messages != []:
    for message in st.session_state.messages[::-1]:
        with st.chat_message(message["role"]):
            st.write(message["content"])



myargs = [
    "Made in India",
    "" " with ❤️ by ",
    link("https://www.linkedin.com/in/anupamisb/", "@Anupam"),
    br(),
    link("https://anupam-purwar.github.io/page/", "SpeeKAR ChatBoT"),
    br(),
    link("https://www.linkedin.com/in/rahul-sundar-311a6977/", "@Rahul"),
    br(),
    link("https://github.com/RahulSundar", "SpeeKAR ChatBoT"),
]


def footer():
    myargs = [
        "Made in India",
        "" " with ❤️ by ",
        link("https://www.linkedin.com/in/anupamisb/", " Anupam for "),
        link("https://anupam-purwar.github.io/page/", "SpeeKAR ChatBoT"),
        ", and",
        link("https://www.linkedin.com/in/rahul-sundar-311a6977/", "@Rahul"),
        link("https://github.com/RahulSundar", "SpeeKAR ChatBoT"),
    ]
    layout(*myargs)


footer()
