from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import openai
from qa import answer_question
#from hugchat import hugchat
#from hugchat.login import Login
import os
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb



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

# App title
st.set_page_config(page_title="🤗💬 I-Venture @ ISB AI-Chat Bot")
st.header("I-Venture @ ISB AI-Chat Bot")

# Hugging Face Credentials
with st.sidebar:
    st.title('🤗💬I-Venture @ ISB Chat Bot')
    st.success('Access to this Gen-AI Powered Chatbot is provided by  [Anupam](https://www.linkedin.com/in/anupamisb/)!!', icon='✅')
    hf_email = 'anupam_purwar2019@pgp.isb.edu'
    hf_pass = 'PASS'
    st.markdown('📖 This app is hosted by I-Venture @ ISB [website](https://i-venture.org/)!')
    image = Image.open('isbdlabs.jpg')
    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
#
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask anything about I-Venture @ ISB ..."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(prompt_input, email, passwd):
     question0=prompt_input
     question=prompt_input
     ans = answer_question(prompt_input)
     # st.write(ans)
     if (ans=='I don\'t know.' or ans=='I don\'t know' or ans== 'I could not find an answer.' or 'I could not find' in ans  or ' I couldn\'t find'  in ans  ):
           question=question0+ " ISB DLabs"
           ans=answer_question(question)
           if (ans=='I don\'t know.'  or ans=='I don\'t know' or ans== 'I could not find an answer.' or 'I could not find' in ans or ' I couldn\'t find'  in ans  ):
             question=question0+ " ISB"
             ans=answer_question(question)
             if (ans=='I don\'t know.'  or ans=='I don\'t know'  or ans== 'I could not find an answer.' or 'I could not find' in ans or ' I couldn\'t find'  in ans  ):
               question=question0+ " I-Venture @ ISB"
               ans=answer_question(question)
               if (ans=='I don\'t know.'  or ans=='I don\'t know'  or ans== 'I could not find an answer.' or 'I could not find' in ans or ' I couldn\'t find'  in ans  ):
                   question=question0+ "Dlabs ISB"
                   ans=answer_question(question)
                   if (ans=='I don\'t know.'  or ans=='I don\'t know'  or ans== 'I could not find an answer.' or 'I could not find' in ans or ' I couldn\'t find'  in ans  ):
                       question=question0+ "Indian School of Business"
                       ans=answer_question(question)
     return ans

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

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt,hf_email,hf_pass) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)


myargs = [
    "Made in India",""
    " with ❤️ by ",
    link("https://www.linkedin.com/in/anupamisb/", "@Anupam"),
     br(),
     link("https://i-venture.org/chatbot/", "ISB ChatBoT"),
    ]

def footer():
    myargs = [
    "Made in India",""
    " with ❤️ by ",
    link("https://www.linkedin.com/in/anupamisb/", " Anupam for "),
    link("https://i-venture.org/chatbot/", "I-Venture @ ISB"),
    ]
    layout(*myargs)
  
footer()
