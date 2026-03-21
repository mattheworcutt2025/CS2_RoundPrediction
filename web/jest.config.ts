export default {
  preset: "ts-jest",
  testEnvironment: "node",
  moduleNameMapper: {
    "^onnxruntime-web$": "<rootDir>/__mocks__/onnxruntime-web.ts",
  },
};
