// Hand landmark detection - uses MediaPipe when available, falls back to simulation

import { useEffect, useRef, useCallback, useState } from "react";
import { Platform } from "react-native";
import { HandDetection } from "@/ml/featureExtractor";

// Try to import MediaPipe
let MediaPipeModule: any = null;
try {
  MediaPipeModule = require("react-native-mediapipe");
} catch (e) {
  // Module not available (web or not installed)
}

export type { HandDetection } from "@/ml/featureExtractor";

export type HandLandmarkFrame = {
  hands: HandDetection[];
  timestamp: number;
};

// Check if running on native platform with MediaPipe
const isNative = Platform.OS === "ios" || Platform.OS === "android";
const isMediaPipeAvailable = isNative && MediaPipeModule !== null;

// Simulated hand detector for web/testing
export function useSimulatedHandDetector(
  enabled: boolean,
  onFrame: (frame: HandLandmarkFrame) => void
) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const frameCountRef = useRef(0);

  useEffect(() => {
    if (!enabled) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    intervalRef.current = setInterval(() => {
      frameCountRef.current++;
      const cycle = Math.floor(frameCountRef.current / 120) % 2;
      
      if (cycle === 1) {
        onFrame({ hands: [], timestamp: Date.now() });
      } else {
        const t = frameCountRef.current * 0.05;
        const createMockLandmarks = () =>
          Array.from({ length: 21 }, (_, i) => ({
            x: 0.3 + Math.sin(t + i * 0.2) * 0.1,
            y: 0.4 + Math.cos(t + i * 0.15) * 0.08,
            z: Math.sin(t * 2 + i * 0.1) * 0.05,
          }));
        
        onFrame({
          hands: [
            { handedness: "Left", landmarks: createMockLandmarks() },
            { handedness: "Right", landmarks: createMockLandmarks() },
          ],
          timestamp: Date.now(),
        });
      }
    }, 66);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [enabled, onFrame]);

  return { isRunning: !!intervalRef.current };
}

// Real MediaPipe hand detector (placeholder - needs frame processor setup)
export function useMediaPipeDetector(
  enabled: boolean,
  onFrame: (frame: HandLandmarkFrame) => void
) {
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!enabled || !isMediaPipeAvailable) return;

    console.log("[MediaPipe] Initializing...");
    // Real initialization would go here
    setIsReady(true);

    return () => {
      setIsReady(false);
    };
  }, [enabled]);

  return { isReady, isAvailable: isMediaPipeAvailable };
}

// Main hook - auto-selects best available detection method
export function useHandLandmarks(
  enabled: boolean,
  onFrame: (frame: HandLandmarkFrame) => void
) {
  const preferReal = isNative; // Prefer real on mobile
  
  const { isReady: realReady, isAvailable: realAvailable } = useMediaPipeDetector(
    enabled && preferReal,
    onFrame
  );

  useSimulatedHandDetector(
    enabled && (!preferReal || !realReady),
    onFrame
  );

  return {
    isUsingRealDetection: realReady,
    isRealAvailable: realAvailable,
  };
}
