class RecorderProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.chunkSize = opts.chunkSize || 8000;
    this.buffer = new Float32Array(this.chunkSize);
    this.index = 0;

    this.port.onmessage = (event) => {
      if (event.data && event.data.flush) {
        if (this.index > 0) {
          const leftover = this.buffer.slice(0, this.index);
          this.port.postMessage(leftover, [leftover.buffer]);
        }
        this.buffer = new Float32Array(this.chunkSize);
        this.index = 0;
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }

    const channelCount = input.length;
    const frames = input[0].length;

    for (let i = 0; i < frames; i++) {
      let sample = 0;
      for (let ch = 0; ch < channelCount; ch++) {
        sample += input[ch][i] || 0;
      }
      sample /= channelCount;

      this.buffer[this.index++] = sample;
      if (this.index >= this.buffer.length) {
        const chunk = this.buffer;
        this.port.postMessage(chunk, [chunk.buffer]);
        this.buffer = new Float32Array(this.chunkSize);
        this.index = 0;
      }
    }

    return true;
  }
}

registerProcessor("recorder-processor", RecorderProcessor);
