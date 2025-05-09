import joblib
import string
import pyttsx3
import speech_recognition as sr

# Load model and vectorizer
model = joblib.load('ivr_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Text-to-speech engine
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Speech recognizer
recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            print(f"User said: {query}")
            return query
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
            speak("Sorry, I didn't catch that.")
            return None
        except sr.RequestError:
            print("❌ Could not request results from the service.")
            speak("Speech recognition service is unavailable.")
            return None

def predict_intent(query):
    query = query.lower().translate(str.maketrans('', '', string.punctuation)).strip()
    vec = vectorizer.transform([query])
    return model.predict(vec)[0]

# Main loop
print("🧠 Voice IVR System (say 'exit' to quit)")
speak("Welcome to the Smart IVR system. Say something...")

while True:
    query = listen()
    if query:
        if 'exit' in query.lower():
            speak("Goodbye!")
            break
        intent = predict_intent(query)
        print(f"🔍 Predicted Intent: {intent}")
        speak(f"Predicted intent is {intent}")
