import openai
import gradio as gr
import os 
from dotenv import load_dotenv
load_dotenv()


#SECRET_TOKEN = os.getenv("SECRET_TOKEN")

SECRET_TOKEN = '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'
messages = [
    {"role": "system", "content": "You are an AI assistant that only gives responses from the website https://i-venture.org/ and you help people make decisions about how to make a difference in others' lives. You also provide the relevant links from that website as part of your answers."},
]

def chatbot(input):
    if input:
        messages.append({"role": "user", "content": input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply

inputs = gr.inputs.Textbox(lines=7, label="Chat with I-venture @ ISB AI powered bot")
outputs = gr.outputs.Textbox(label="Reply")



gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="Talk with I-venture @ ISB",
             description="Anything you want to find out about entreprenuership at ISB. Sample questions include >>> how to get incubated at ISB Dlabs? >>> What is the latest event being organized by I-venture @ ISB? >>> and more",
             theme="compact", live=True,).launch(share=True)    

# , debug=True
