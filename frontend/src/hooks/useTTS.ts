import * as Speech from "expo-speech";
import { useCallback, useEffect, useRef, useState } from "react";
import { Platform } from "react-native";

// Thin hook over expo-speech with three layers of resilience:
// 1. Picks the best available Urdu voice on first use (ur-PK > ur > hi-IN as
//    a last-ditch fallback because hi-IN tends to ship on more devices and
//    can render Urdu glyphs intelligibly).
// 2. Catches every error so a failing speak() never crashes the screen.
// 3. Reports a `supported` flag the UI can use to show a hint when speech
//    isn't available (e.g. plain web with no SpeechSynthesis).
export function useTTS() {
  const [supported, setSupported] = useState(true);
  const [voice, setVoice] = useState<string | undefined>(undefined);
  const speakingRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // expo-speech always exists, but the underlying engine may not on web.
        if (Platform.OS === "web" && typeof window !== "undefined") {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const synth = (window as any).speechSynthesis;
          if (!synth) {
            if (!cancelled) setSupported(false);
            return;
          }
        }
        const voices = await Speech.getAvailableVoicesAsync();
        if (cancelled) return;
        // Prefer Urdu, then Hindi (similar phonetics + more devices ship it),
        // then English as a final fallback.
        const preferred =
          voices.find((v) => v.language?.toLowerCase().startsWith("ur")) ??
          voices.find((v) => v.language?.toLowerCase().startsWith("hi")) ??
          voices.find((v) => v.language?.toLowerCase().startsWith("en"));
        setVoice(preferred?.identifier);
      } catch {
        if (!cancelled) setSupported(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const speak = useCallback(
    async (text: string, opts?: { language?: string }) => {
      if (!text?.trim()) return;
      try {
        const isSpeaking = await Speech.isSpeakingAsync().catch(() => false);
        if (isSpeaking) Speech.stop();
        speakingRef.current = true;
        Speech.speak(text, {
          language: opts?.language ?? "ur-PK",
          voice,
          rate: 0.85,
          pitch: 1.0,
          onDone: () => {
            speakingRef.current = false;
          },
          onStopped: () => {
            speakingRef.current = false;
          },
          onError: () => {
            speakingRef.current = false;
          },
        });
      } catch {
        // Last-resort: silently swallow so a missing TTS engine never crashes.
      }
    },
    [voice],
  );

  const stop = useCallback(() => {
    Speech.stop();
    speakingRef.current = false;
  }, []);

  return { speak, stop, supported, voice };
}
