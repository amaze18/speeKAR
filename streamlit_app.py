from collections import namedtuple
import altair as alt

import os, time
import pandas as pd
import math
import glob
import base64
from io import StringIO
import boto3

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
    return a(_href=link, _target="_blank", style=styles(**style))


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
from rouge import Rouge
import time

def calculate_rouge_scores(answer,context):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(answer,context)
    return rouge_scores


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

# Initialize session state variables
if "uploaded_status" not in st.session_state:
    st.session_state["uploaded_status"] = False
if "query_counter" not in st.session_state:
    st.session_state["query_counter"] = 0
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader(label = "")
if st.session_state["uploaded_status"] == False and uploaded_file is not None:
    create_db.clear()
    readdoc_splittext.clear()
    readdoc_splittext_pdf.clear()

if (uploaded_file is not None):
    st.session_state["uploaded_status"] = True
elif uploaded_file is None:
    st.session_state["uploaded_status"] = False
    st.session_state["query_counter"] = 0
    create_db.clear()
    readdoc_splittext.clear()
    readdoc_splittext_pdf.clear()
    st.write("You can upload your document now.")


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

        # Display the chat input box
        query_text = st.chat_input("Let me know what you have in mind!")

        # Check if the user has entered a query
        if query_text != "":
            # Add the user message to the messages list
            st.session_state.messages.append({"role": "user", "content": query_text})

            # Display the user message in the chat message container
            with st.chat_message("user"):
                st.markdown(query_text)

            # Generate a response from the chatbot
            with st.chat_message("assistant"):
                # Your existing code to generate a response from the chatbot
                # ...

                # Add the assistant's response to the messages list
                st.session_state.messages.append({"role": "assistant", "content": ans})

                # Display the assistant's response in the chat message container
                st.markdown(ans)

    # At the end of the script
    # Loop through all messages and display them
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
