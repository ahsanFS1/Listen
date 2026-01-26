import edge_tts
import asyncio
import os

async def _speak(text: str, voice: str, save_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(save_path)
    os.system(f"afplay {save_path}")

def speak(text: str, lang: str = "en", save_path: str = "output.mp3"):
    voices = {
        "en": "en-US-GuyNeural",
        "ur": "ur-PK-AsadNeural"
    }
    asyncio.run(_speak(text, voices[lang], save_path))


if __name__ == "__main__":
    speak("Chup kr be kuttay", lang="en")
    # speak("آپ کیسے ہیں؟", lang="ur")