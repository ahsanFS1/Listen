import { useEffect, useRef, useState, useCallback } from "react";
import { SignPipeline, Prediction, PipelineMode } from "@/ml/pipeline";
import { useHandLandmarks, HandLandmarkFrame } from "./useHandLandmarks";

// Drives the pipeline only while `active` is true. Predictions stop
// flowing the moment the camera is paused, so the Translate screen never
// commits ghost words while the signer's hand is down.
export function useSignRecognition(
  active: boolean,
  mode: PipelineMode = "words",
  isHoldingSign: boolean = false
) {
  const pipelineRef = useRef<SignPipeline | null>(null);
  if (!pipelineRef.current) pipelineRef.current = new SignPipeline();

  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [useRealDetection, setUseRealDetection] = useState(false);

  // Update mode when it changes
  useEffect(() => {
    const pipeline = pipelineRef.current;
    if (pipeline) {
      pipeline.setMode(mode);
    }
  }, [mode]);

  // Handle hand detection frames for real mode
  const handleHandFrame = useCallback(async (frame: HandLandmarkFrame) => {
    const pipeline = pipelineRef.current;
    if (!pipeline || !active) return;
    
    const p = await pipeline.processFrame(frame.hands);
    setPrediction(p);
  }, [active]);

  // Use hand detection hook - always runs, handles real vs simulated internally
  const { isUsingRealDetection, isRealAvailable } = useHandLandmarks(
    active && useRealDetection, 
    handleHandFrame
  );

  // Mock mode - runs when real detection is not enabled
  const [simulateHands, setSimulateHands] = useState(true); // Toggle for testing
  
  useEffect(() => {
    // If using real hand detection, don't run mock
    if (useRealDetection || isUsingRealDetection) return;
    
    const pipeline = pipelineRef.current!;
    if (active) {
      pipeline.start();
      const id = setInterval(() => {
        // Mock mode: only predict when holding the sign button
        const p = pipeline.mockTick(simulateHands && isHoldingSign);
        setPrediction(p);
      }, 80);
      return () => {
        clearInterval(id);
        pipeline.stop();
        setPrediction(null);
      };
    }
    pipeline.stop();
    setPrediction(null);
    return undefined;
  }, [active, useRealDetection, simulateHands, isUsingRealDetection, handleHandFrame, isHoldingSign]);

  const reset = useCallback(() => {
    pipelineRef.current?.reset();
    setPrediction(null);
  }, []);

  // Toggle between mock and real detection (for development/testing)
  const toggleDetectionMode = useCallback(() => {
    setUseRealDetection(prev => !prev);
  }, []);

  // Check if hands are visible - prediction will be null or hasHands=false when no hands
  const hasHands = prediction?.hasHands ?? false;
  const isCommitted = prediction?.committed ?? false;

  // Toggle hand simulation for web testing
  const toggleHandSimulation = useCallback(() => {
    setSimulateHands(prev => !prev);
  }, []);

  return { 
    prediction, 
    reset, 
    hasHands, 
    isCommitted,
    useRealDetection,
    toggleDetectionMode,
    isRealAvailable,
    simulateHands,
    toggleHandSimulation,
  };
}
