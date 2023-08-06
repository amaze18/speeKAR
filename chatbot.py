import gradio as gr

messages = [
    {"role": "system", "content": "You are an AI assistant that only gives responses from the website https://i-venture.org/ and you help people make decisions about how to make a difference in others' lives. You also provide the relevant links from that website as part of your answers."},
]

def chatbot(input):
    if input:
        context = create_context(input, df)
        message=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {input}\nAnswer:",
        messages.append({"role": "user", "content": message})
        chat = openai.ChatCompletion.create(
            temperature=0.5, model="gpt-3.5-turbo", messages=messages,
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply

inputs = gr.inputs.Textbox(lines=7, label="Chat with I-venture @ ISB AI powered bot")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="Talk with I-venture @ ISB",
             description="Anything you want to find out about entreprenuership at ISB. Sample questions include >>> how to get incubated at ISB Dlabs? >>> What is the team behind  I-venture @ ISB? >>> and more",
             theme="compact").launch(share=True, debug=True)