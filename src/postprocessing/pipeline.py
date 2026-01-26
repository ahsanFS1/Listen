from grammar import fix_grammar, translate_to_urdu
from tts import speak

def run(text: str, output_lang: str = "en"):
    print(f"Raw:    {text}")
    
    fixed = fix_grammar(text)
    print(f"Fixed:  {fixed}")
    
    if output_lang == "ur":
        final = translate_to_urdu(fixed)
        print(f"Urdu:   {final}")
    else:
        final = fixed
    
    speak(final, lang=output_lang)


if __name__ == "__main__":
    test_sentences = [
        # Basic broken grammar
        "i go store yesterday buy milk",
        
        # Missing articles/prepositions
        "she give me book read",
        "he sit chair wait long time",
        
        # Word order issues (common in sign language)
        "yesterday night dinner what you eat",
        "homework finish you when",
        "movie good very i like",
        
        # Missing verbs/connectors
        "my friend sick hospital now",
        "tomorrow meeting important boss angry",
        
        # Tense confusion
        "yesterday i go park see bird fly",
        "next week she come visit stay three day",
        
        # Complex ideas, broken structure
        "mother say clean room before go outside play friend",
        "teacher explain but understand nothing i confused",
        "rain outside umbrella forget home wet now",
        
        # Emotional/expressive
        "why you late always me wait angry",
        "surprise party him happy cry",
        
        # Questions
        "where you go yesterday night",
        "how many people come party",
    ]
    
    for sentence in test_sentences:
        run(sentence, output_lang="ur")
        print("-" * 40)