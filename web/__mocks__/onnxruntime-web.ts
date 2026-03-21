export const env = { wasm: { numThreads: 1 } };
export class InferenceSession {
  static async create() { return new InferenceSession(); }
  async run() { return { probability: { data: new Float32Array([0.5]) } }; }
}
export class Tensor {
  constructor(public type: string, public data: Float32Array, public dims: number[]) {}
}
