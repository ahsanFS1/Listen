import os
from dotenv import load_dotenv
from groq import Groq
from deep_translator import GoogleTranslator

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def fix_grammar(text: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are a grammar corrector. Fix grammar and make sentences natural English.

RULES:
- Return ONLY the corrected sentence
- Do NOT add new meaning or information
- Do NOT change who is doing what
- Do NOT answer questions, just fix their grammar
- Keep the original meaning exactly
- If unsure about meaning, keep it simple

Example:
Input: "tomorrow meeting important boss angry"
Output: "Tomorrow there is an important meeting. The boss is angry."

Input: "i go store yesterday buy milk"
Output: "I went to the store yesterday to buy milk."

Input: "she happy see friend"
Output: "She is happy to see her friend."
"""
            },
            {
                "role": "user", 
                "content": text
            }
        ]
    )
    return response.choices[0].message.content
def translate_to_urdu(text: str) -> str:
    return GoogleTranslator(source='en', target='ur').translate(text)


if __name__ == "__main__":
    text = "i go store yesterday buy milk"
    fixed = fix_grammar(text)
    urdu = translate_to_urdu(fixed)
    print(f"Input:  {text}")
    print(f"Fixed:  {fixed}")
    print(f"Urdu:   {urdu}")