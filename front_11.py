import gradio as gr
from chat_ai_10 import chat_user

demo = gr.Interface(fn=chat_user, inputs="textbox", outputs="textbox")

demo.launch(share=True)
