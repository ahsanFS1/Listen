import { useEffect, useRef, useState } from "react";
import { SignPipeline, Prediction } from "@/ml/pipeline";

// Drives the pipeline only while `active` is true. Predictions stop
// flowing the moment the camera is paused, so the Translate screen never
// commits ghost words while the signer's hand is down.
export function useSignRecognition(active: boolean) {
  const pipelineRef = useRef<SignPipeline | null>(null);
  if (!pipelineRef.current) pipelineRef.current = new SignPipeline();

  const [prediction, setPrediction] = useState<Prediction | null>(null);

  useEffect(() => {
    const pipeline = pipelineRef.current!;
    if (active) {
      pipeline.start();
      const id = setInterval(() => {
        const p = pipeline.mockTick();
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
  }, [active]);

  const reset = () => {
    pipelineRef.current?.reset();
    setPrediction(null);
  };

  return { prediction, reset };
}
