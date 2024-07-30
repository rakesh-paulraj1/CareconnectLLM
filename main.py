import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
import speech_recognition as sr
import playsound  # Alternative to mpg321 for cross-platform compatibility

# Load the Medical LLaMA model
def load_medical_llama_model(token):
    model_name = "ruslanmv/Medical-Llama3-8B"  # Ensure this is the correct model identifier
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    return tokenizer, model

# Generate response using the model
def generate_response(tokenizer, model, input_text):
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text

# Convert text to speech
def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    playsound.playsound(filename)

# Convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=5)
        print("Listening...")
        audio = recognizer.listen(source)
    
    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
    return None

if __name__ == "__main__":
    input_text = speech_to_text()
    if input_text:
        hf_token = "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"  # Replace with your actual token
        tokenizer, model = load_medical_llama_model(hf_token)
        response_text = generate_response(tokenizer, model, input_text)
        print(f"Response: {response_text}")
        text_to_speech(response_text)
