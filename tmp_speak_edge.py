def speak_word_edge(word_letters):
    """
    Speak the Urdu word using Microsoft Edge Text-to-Speech.
    
    Args:
        word_letters: String of letter names (e.g., "Alif Chay Dochahay Alif")
    """
    if not TTS_SUPPORT:
        return
    
    # Convert letter names to Urdu script
    urdu_chars = []
    for letter in word_letters.strip().split():
        if letter in PSL_TO_URDU:
            urdu_chars.append(PSL_TO_URDU[letter])
    
    if not urdu_chars:
        return
    
    # Join to create Urdu word
    urdu_word = "".join(urdu_chars)
    
    # Speak in a separate thread to avoid blocking
    import threading
    import tempfile
    import os
    
    def speak():
        try:
            # Create async function to generate speech with edge-tts
            async def generate_speech():
                communicate = edge_tts.Communicate(urdu_word, VOICE)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_file = fp.name
                await communicate.save(temp_file)
                return temp_file
            
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            temp_file = loop.run_until_complete(generate_speech())
            loop.close()
            
            # Play the audio
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            pygame.mixer.music.unload()
            os.remove(temp_file)
        except Exception as e:
            print(f"TTS Error: {e}")
    
    thread = threading.Thread(target=speak, daemon=True)
    thread.start()
