import time, os
import io
import scipy
import numpy as np

#----------------SPEECH RECOGNITION /AUDIO/TEXT TO SPEECH DEPENDENCIES-------------#
from scipy.io.wavfile import read as wav_read
from pydub import AudioSegment
from pydub.playback import play
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
import speech_recognition as sr
import ffmpeg
from gtts import gTTS
import soundfile as sf

#----STREAMLIT RELATED----#
import streamlit as st
import requests
import re
import urllib.request


#---------DOCUMENT/WEBSITE PARSING---------#
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse

#-------DATA FRAME/DOCX/TEXT HANDLING----------$
import pandas as pd
import pprint as pp
from docx import Document
from docx.shared import Inches
import textwrap
import glob

#-----------------------------------------------------------------------------------------#
#---------------------------------OPENAI and LANGCHAIN DEPENDENCIES-----------------------#
#-----------------------------------------------------------------------------------------#
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


#--------------KEYBERT MODEL ---------------#
from keybert import KeyBERT


#----------DEFAULTS:------------#
LANGUAGE = 'en'


#---------------READ THE UPLOADED DOCUMENT AND GENERATE THE SPLIT---------------# 
def readdoc_splittext(filename):
    """
    This functions takes in an input document and finds the headings, and
    splits them based on the chunks needed. 
    """
    document = Document(filename)
    headings = []
    para_texts = []
    i=0
    j=0
    n = 1500 #Number of characters to be included in a single chunk of text  
    t=''
    for paragraph in document.paragraphs:

        if paragraph.style.name == "Heading 2":
            no_free_text = "".join(filter(lambda x: not x.isdigit(), paragraph.text))
            k = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

            getVals = list(filter(lambda x: x in k, no_free_text))
            result = "".join(getVals)
            headings.append(result.lower())
            i=1
            #print(paragraph.text)

        elif paragraph.style.name == "normal" and i==1   :
            t+=paragraph.text


            try:
              if(document.paragraphs[j+1].style.name=="Heading 2"):
                i=0
                # to remove numeric digits from string

                para_texts.append(t)
                t=''
                #print(i)
            except:
              print("reached doc end")
              para_texts.append(t)
              #print(i)
        j+=1

    print(len(para_texts), len(headings))
    for h, t in zip(headings, para_texts):
        print(h, t)

    all_text=''
    for text in para_texts:
      all_text+=text

    line = all_text
    text_split=[line[i:i+n] for i in range(0, len(line), n)]
    i=0
    for t in  para_texts:
        print(i)
        print(t)
        i+=1
    return all_text, text_split
    
    
#all_text, text_split = readdoc_splittext()


def remove_newlines(serie):
    serie = serie.replace('\n', ' ')
    serie = serie.replace('\\n', ' ')
    serie = serie.replace('  ', ' ')
    serie = serie.replace('  ', ' ')
    return serie

#----------------CREATE CONTEXT-----------------------#
def create_context(query, text_split):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    # Get the distances from the embeddings
    #i=0
    kw_model = KeyBERT()

    sentences = text_split #[i for i in nlp(text).sents]

    returns =[]
    keystart= time.time()
    keywords_q =[]


    keywords_query = kw_model.extract_keywords(query)
    keywords = []

    for j in range(len(keywords_query)):

          if(keywords_query[j][1]> 0.35 ):
            if( keywords_query[j][0] not in keywords_q):
              #print(keywords[j][0], keywords[j][1])
              keywords_q.append(keywords_query[j][0])


    i=0
    keyword_doc={}
    keyend= time.time()

    for sent in sentences:
      if(isinstance(sent, str) and len(sent)>6):
        keywords = kw_model.extract_keywords(sent)
        keyword_doc_sent=[]

        for j in range(len(keywords)):
            # Add the heading of  the para corresponding to the sentence
            if(j ==0):
              keyword_doc_sent.append(keywords[j][0])
              for h, pt in zip(headings, para_texts):

                      if(sent in pt):
                          keyword_doc_sent.append(h)
                          #print("sent ::",sent)
                          #print("heading::", h)
                          #print("para text ::",pt)
            if(keywords[j][1]> 0.35): # and keywords[j][0] not in keyword_doc ):
               if(keywords[j][0] not in keyword_doc_sent ):
                 keyword_doc_sent.append(keywords[j][0])
      keyword_doc[i]=keyword_doc_sent

      i+=1

    #print("doc keywords:: ",keyword_doc)
    #print("q keywords::",keywords_q)
    search_start= time.time()

    for i in range(len(keyword_doc)):
          for k in range(len(keywords_q)):
                  match_count= 0
                  if(keywords_q[k] in keyword_doc[i]):
                    match_count+=1
                    keywords.append(keywords_q[k])
                    #print(keywords_q[k],keyword_doc[i] )
                    #print("match_count::",match_count)
                    if( (match_count>=1 or match_count>=len(keywords_q)  ) ):
                      #print("Document matched :",i, "::")
                      if(remove_newlines(text_split[i]) not in returns ):
                        #context_q+=remove_newlines(sent)
                        returns.append(remove_newlines(text_split[i]))
                        #print(returns,match_count )


    searchend= time.time()
    search_time = searchend-search_start

    cur_len = 0


    # Return the context
    return "\n\n###\n\n".join(returns), keywords

#------------------------SLIM KAR BASED CHATBOT----------------------------#
def chatbot_slim(query, text_split):
    """
    Here, this function takes in the textual query, along with the textual context and uses KAR framework to geerate a suitable response 
    with little to almost no hallucinations. Here, openai's davnci-003 has been used to generate the response. 
    """
    if input:

        stime= time.time()

        context, keywords = create_context(query, text_split)

        ctype=['stuff', 'map_reduce', 'refine', 'map_rerank']
        template= '''
              You are a helpful assistant who answers question based on context provided: {context}

              If you don't have enough information to answer the question, say: "Sorry, I cannot answer that".

              '''
        template= '''
                  You are a helpful assistant who answers question based on context provided: {context}

                  If you don't have enough information to answer the question, say: "I cannot answer".

                  '''
        template= ''' You answer question based on context below, and if question can't be answered based on context, say \"I don't know\"\n\nContext: {context} '''

        system_message_prompt= SystemMessagePromptTemplate.from_template(template)

        #Human question prompt

        human_template= 'Answer following question: {question}'

        template= ''' Answer question {question} based on context below, and if question can't be answered based on context,
        say \"I don't know\"\n\nContext: {context}

        Answer:
        '''

        template= ''' Use following pieces of context to answer the question. Provide answer in full detail using provided context.
        If you don't know the answer, say I don't know
        {context}
        Question : {question}
        Answer:'''


        human_message_prompt= HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt= ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt])

        chunk_size = 1024
        PROMPT = PromptTemplate(input_variables=["context", "question"],  template=template)

        chain_type_kwargs = {"prompt": PROMPT}

        question = query

        openai.api_key = "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"
        model="text-davinci-003"
        chat  = openai.Completion.create(
            prompt=f"You answer question based on context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            model=model,
        )

        reply = chat["choices"][0]["text"].strip() # ['choices'][0]['message']['content']
        messages.append({"role": "assistant", "content": reply})

        return reply, context, keywords

#----------------TEXT TO SPEECH FUNCTION FOR ANSWER READOUT---------#
def texttospeech_raw(text,language,savename="answer",slow=False):
  """
  This function here, calls the google text to speech engine to read out the answer generated by the KAR framework. 
  """
  myobj = gTTS(text=text, lang=language, slow=False)

  # Saving the converted audio in a mp3 file
  myobj.save(savename+".mp3")
  sound = AudioSegment.from_mp3(savename + ".mp3")
  sound.export(answer+".wav", format="wav")


#---------------SPEECH RECOGNITION---------#
def speechtotext(query_audio):
  """
  This function takes in a ".wav" audio file as input and converts into text. 
  """  
  r = sr.Recognizer()

  audio_ex = sr.AudioFile(query_audio)
  type(audio_ex)

  # Create audio data
  with audio_ex as source:
      audiodata = r.record(audio_ex)
  type(audiodata)
  # Extract text
  text = r.recognize_google(audio_data=audiodata, language='en-US')
  return text
