import * as Speech from "expo-speech";
import { useCallback, useEffect, useRef, useState } from "react";
import { Platform } from "react-native";

// TTS Hook with Android-specific fixes
// 1. Better voice selection for Urdu on Android
// 2. Retries with fallbacks when primary voice fails
// 3. Queue management to prevent overlapping speech
export function useTTS() {
  const [supported, setSupported] = useState(true);
  const [voice, setVoice] = useState<Speech.Voice | null>(null);
  const [voices, setVoices] = useState<Speech.Voice[]>([]);
  const speakingRef = useRef(false);
  const queueRef = useRef<string[]>([]);

  useEffect(() => {
    let cancelled = false;
    
    const initTTS = async () => {
      try {
        // Check if speech is available
        const isAvailable = await Speech.isSpeakingAsync().then(() => true).catch(() => false);
        
        if (Platform.OS === "web" && typeof window !== "undefined") {
          const synth = (window as unknown as { speechSynthesis?: SpeechSynthesis }).speechSynthesis;
          if (!synth) {
            if (!cancelled) setSupported(false);
            return;
          }
        }
        
        // Get available voices
        const availableVoices = await Speech.getAvailableVoicesAsync();
        if (cancelled) return;
        
        setVoices(availableVoices);
        
        // Find best voice for Urdu
        // Priority: ur-PK > ur > hi-IN > hi > en
        const urduVoice = availableVoices.find((v) => 
          v.language?.toLowerCase() === "ur-pk" || 
          v.language?.toLowerCase().startsWith("ur")
        );
        
        const hindiVoice = availableVoices.find((v) => 
          v.language?.toLowerCase().startsWith("hi")
        );
        
        const englishVoice = availableVoices.find((v) => 
          v.language?.toLowerCase().startsWith("en")
        );
        
        const bestVoice = urduVoice ?? hindiVoice ?? englishVoice ?? availableVoices[0];
        
        if (bestVoice) {
          setVoice(bestVoice);
          console.log(`[TTS] Selected voice: ${bestVoice.identifier} (${bestVoice.language})`);
        } else {
          console.warn("[TTS] No suitable voice found");
        }
        
        setSupported(true);
      } catch (err) {
        console.error("[TTS] Initialization error:", err);
        if (!cancelled) setSupported(false);
      }
    };
    
    initTTS();
    
    return () => {
      cancelled = true;
      Speech.stop();
    };
  }, []);

  // Process speech queue
  const processQueue = useCallback(async () => {
    if (speakingRef.current || queueRef.current.length === 0) return;
    
    const text = queueRef.current.shift();
    if (!text) return;
    
    speakingRef.current = true;
    
    try {
      const options: Speech.SpeechOptions = {
        language: voice?.language ?? "ur-PK",
        voice: voice?.identifier,
        rate: Platform.OS === "android" ? 0.9 : 0.85,
        pitch: 1.0,
        onDone: () => {
          speakingRef.current = false;
          // Process next in queue
          setTimeout(() => processQueue(), 100);
        },
        onStopped: () => {
          speakingRef.current = false;
        },
        onError: (error) => {
          console.error("[TTS] Speech error:", error);
          speakingRef.current = false;
          // Retry with fallback voice
          if (voice && voices.length > 1) {
            const fallback = voices.find(v => v.identifier !== voice.identifier);
            if (fallback) {
              setVoice(fallback);
              queueRef.current.unshift(text);
            }
          }
          setTimeout(() => processQueue(), 100);
        },
      };
      
      Speech.speak(text, options);
    } catch (err) {
      console.error("[TTS] Speak error:", err);
      speakingRef.current = false;
    }
  }, [voice, voices]);

  const speak = useCallback(
    async (text: string, opts?: { language?: string; priority?: boolean }) => {
      if (!text?.trim()) return;
      if (!supported) {
        console.warn("[TTS] Speech not supported");
        return;
      }
      
      // Stop current speech if priority
      if (opts?.priority) {
        Speech.stop();
        speakingRef.current = false;
        queueRef.current = [];
      }
      
      // Add to queue
      queueRef.current.push(text);
      
      // Start processing if not already speaking
      if (!speakingRef.current) {
        processQueue();
      }
    },
    [supported, processQueue]
  );

  const stop = useCallback(() => {
    Speech.stop();
    speakingRef.current = false;
    queueRef.current = [];
  }, []);

  const isSpeaking = useCallback(async () => {
    try {
      return await Speech.isSpeakingAsync();
    } catch {
      return false;
    }
  }, []);

  return { 
    speak, 
    stop, 
    supported, 
    voice: voice?.identifier,
    isSpeaking,
    voiceLanguage: voice?.language ?? "ur-PK",
  };
}
