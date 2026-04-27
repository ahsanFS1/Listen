// Real MediaPipe Hand Detection for iOS/Android
// Uses react-native-mediapipe for actual hand landmark detection

import { useEffect, useRef, useCallback, useState } from "react";
import { Platform } from "react-native";
import { HandDetection } from "@/ml/featureExtractor";

// Lazy import to avoid crashes if module isn't available
let MediaPipe: typeof import("react-native-mediapipe") | null = null;

try {
  MediaPipe = require("react-native-mediapipe");
} catch (e) {
  console.warn("[MediaPipe] Native module not available:", e);
}

export type HandLandmarkFrame = {
  hands: HandDetection[];
  timestamp: number;
};

// Check if MediaPipe is available on this platform
export function isMediaPipeAvailable(): boolean {
  return Platform.OS !== "web" && MediaPipe !== null;
}

// Initialize MediaPipe hand detector
export function useMediaPipeHands(
  enabled: boolean,
  onFrame: (frame: HandLandmarkFrame) => void
) {
  const [isInitialized, setIsInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const detectorRef = useRef<any>(null);

  useEffect(() => {
    if (!enabled || !isMediaPipeAvailable()) {
      return;
    }

    const initDetector = async () => {
      try {
        if (!MediaPipe) {
          throw new Error("MediaPipe module not loaded");
        }

        // NOTE: react-native-mediapipe@0.6.0 does NOT export a HandLandmarker
        // (only object/pose/face detection). Real hand landmark detection
        // requires either a newer version of this package or a custom
        // vision-camera frame processor that calls MediaPipe Tasks natively.
        // Until that's wired, this hook is a no-op and the app falls back
        // to the simulated detector in `useHandLandmarks.ts`.
        throw new Error(
          "HandLandmarker not available in react-native-mediapipe@0.6.0",
        );
      } catch (e) {
        console.error("[MediaPipe] Failed to initialize:", e);
        setError(e instanceof Error ? e.message : "Unknown error");
      }
    };

    initDetector();

    return () => {
      if (detectorRef.current) {
        detectorRef.current.close();
        detectorRef.current = null;
      }
    };
  }, [enabled]);

  // Process frame - this would be called from camera frame processor
  const processFrame = useCallback(
    async (imageSource: any) => {
      if (!detectorRef.current || !enabled) return;

      try {
        const results = await detectorRef.current.detect(imageSource);
        
        if (results && results.handedness && results.landmarks) {
          const hands: HandDetection[] = results.handedness.map(
            (handedness: any, index: number) => ({
              handedness: handedness[0].categoryName as "Left" | "Right",
              landmarks: results.landmarks[index].map((lm: any) => ({
                x: lm.x,
                y: lm.y,
                z: lm.z || 0,
              })),
            })
          );

          onFrame({
            hands,
            timestamp: Date.now(),
          });
        } else {
          // No hands detected
          onFrame({ hands: [], timestamp: Date.now() });
        }
      } catch (e) {
        console.error("[MediaPipe] Detection error:", e);
      }
    },
    [enabled, onFrame]
  );

  return {
    isInitialized,
    isAvailable: isMediaPipeAvailable(),
    error,
    processFrame,
  };
}
