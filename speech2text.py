import speech_recognition as sr
from transformers import pipeline

def speech_to_text():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=5)
        print("Listening...")
        audio = recognizer.listen(source)
    
    try:
        # Recognize speech using Google Web Speech API
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")
        return None

def analyze_sentiment(text):
    # Load sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    # Analyze sentiment
    result = sentiment_pipeline(text)
    return result

if __name__ == "__main__":
    text = speech_to_text()
    if text:
        sentiment_result = analyze_sentiment(text)
        print(f"Sentiment analysis: {sentiment_result}")
