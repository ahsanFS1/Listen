import { F_DIM, T_WINDOW } from "./constants";

// Ring-style rolling buffer matching the desktop `_push_frame` behaviour.
// Storage is a single flat Float32Array of size T_WINDOW * F_DIM so the
// window copy in classify() is a single TypedArray clone.
export class RollingBuffer {
  private readonly buffer: Float32Array;
  private _fillCount = 0;

  constructor(
    private readonly windowSize: number = T_WINDOW,
    private readonly featureDim: number = F_DIM,
  ) {
    this.buffer = new Float32Array(this.windowSize * this.featureDim);
  }

  get isFull(): boolean {
    return this._fillCount >= this.windowSize;
  }

  get fillCount(): number {
    return this._fillCount;
  }

  push(frame: Float32Array): void {
    if (frame.length !== this.featureDim) {
      throw new Error(
        `RollingBuffer.push: frame length ${frame.length} != featureDim ${this.featureDim}`,
      );
    }
    const total = this.windowSize * this.featureDim;
    // Shift left by one row
    this.buffer.copyWithin(0, this.featureDim, total);
    // Write new frame at the tail
    this.buffer.set(frame, total - this.featureDim);
    if (this._fillCount < this.windowSize) this._fillCount += 1;
  }

  snapshot(): Float32Array {
    return this.buffer.slice();
  }

  reset(): void {
    this.buffer.fill(0);
    this._fillCount = 0;
  }
}
