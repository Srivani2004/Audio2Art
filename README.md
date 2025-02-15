# Audio2Art

****TEXT TOIMAGE GENERATION**
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
login(token=" ")
from diffusers import StableDiffusionPipeline, DDPMPipeline
pipe = StableDiffusionPipeline.from_pretrained("ZB-Tech/Text-to-Image", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt1 = "white horse"
image1 = pipe(prompt1).images[0]

image1.save("image0.png")
image1


**INTERFACE**
import gradio as gr
import openai
import os
import warnings
warnings.filterwarnings("ignore")

openai.api_key=""

def chatgpt_api(input_text):
  messages=[
  {"role":"system","content":"You are a kind helpful assistant."}]

  if input_text:
    messages.append(
        {"roles":"user","content": 'summarize this text "{}" into a short and concise Dall-e2 prompt'}
        )
    chat_completion=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",messages=messages
    )
  reply=chat_completion.choices[0].message.content
  return reply

def dall_e_api(dalle_prompt):
  dalle_response=openai.Image.create(
      prompt=dalle_prompt,
      size="512x512"
  )
  return dalle_response['data'][0]['url']

  def whisper_transcribe(audio):
  os.rename(audio, audio + '.wav')
  audio_file=open(audio + '.wav', "rb")
  transcript=openai.Audio.transcribe("whisper-1",audio_file)
  dalle_prompt=chatgpt_api(transcript["text"])
  return transcript["text"], dall_e_api(dalle_prompt)

  output_1=gr.Textbox(label="Speech to Text")
output_2=gr.Image(label="DALL-E Image")

speech_interface = gr.Interface(
    fn=whisper_transcribe,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),  # Replace 'source' with 'sources' and provide a list
    outputs=[output_1, output_2],
    title="Generate Images using Voice"
)
speech_interface.launch(debug=True)
  
