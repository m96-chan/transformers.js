import { jest } from "@jest/globals";

import { ChatterboxModel, PreTrainedModel, Tensor, env, LogLevel } from "../../../src/transformers.js";

import { MAX_TEST_EXECUTION_TIME } from "../../init.js";

// Helper function to create a model with only a fake `conditional_decoder` session, so no weights are loaded
function createModel(run) {
  const model = Object.create(ChatterboxModel.prototype);
  model.sessions = {
    conditional_decoder: { inputNames: ["speech_tokens", "speaker_features", "speaker_embeddings"], run },
  };
  return model;
}

// Helper function to stub the outputs of `super.generate`, returning the KV cache handed to the subclass
function mockSuperGenerate() {
  const past_key_values = { dispose: jest.fn(async () => {}) };
  jest.spyOn(PreTrainedModel.prototype, "generate").mockResolvedValue({
    sequences: new Tensor("int64", [0n, 1n, 2n, 3n, 4n], [1, 5]),
    audio_tokens: new Tensor("int64", [10n, 11n], [1, 2]),
    speaker_embeddings: new Tensor("float32", new Float32Array(4), [1, 4]),
    speaker_features: new Tensor("float32", new Float32Array(4), [1, 4]),
    past_key_values,
  });
  return past_key_values;
}

export default () => {
  describe("ChatterboxModel", () => {
    describe("generate", () => {
      const input_ids = new Tensor("int64", [0n], [1, 1]);
      const original_log_level = env.logLevel;

      // The rejecting session is logged by `sessionRun`, so silence it to keep the test output clean
      beforeAll(() => {
        env.logLevel = LogLevel.NONE;
      });

      afterAll(() => {
        env.logLevel = original_log_level;
      });

      afterEach(() => {
        jest.restoreAllMocks();
      });

      it(
        "disposes the KV cache",
        async () => {
          const past_key_values = mockSuperGenerate();
          const model = createModel(async () => ({ waveform: new Tensor("float32", new Float32Array(4), [1, 4]) }));

          const waveform = await model.generate({ input_ids });

          expect(waveform.dims).toEqual([1, 4]);
          expect(past_key_values.dispose).toHaveBeenCalledTimes(1);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "disposes the KV cache when the conditional decoder rejects",
        async () => {
          const past_key_values = mockSuperGenerate();
          const model = createModel(async () => {
            throw new Error("conditional_decoder failed");
          });

          await expect(model.generate({ input_ids })).rejects.toThrow("conditional_decoder failed");

          expect(past_key_values.dispose).toHaveBeenCalledTimes(1);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });
  });
};
