import {accessSync, constants, statSync} from "node:fs";
import {resolve} from "node:path";

export interface LocalInferenceConfig {
  host: string;
  port: number;
  embeddingEnabled: boolean;
  embeddingModelPath: string;
  embeddingModelName: string;
  embeddingDimensions: number;
  embeddingMaxTokens: number;
  rerankerEnabled: boolean;
  rerankerModelPath: string;
  rerankerModelName: string;
  rerankerMaxTokens: number;
  cudaVisibleDevices: string;
}

function positiveInteger(name: string, fallback: number): number {
  const raw = process.env[name];
  const value = raw === undefined ? fallback : Number.parseInt(raw, 10);
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

export function loadConfig(): LocalInferenceConfig {
  const embeddingEnabled = process.env.PAPEROS_EMBEDDING_ENABLED !== "false";
  const modelPath = embeddingEnabled
    ? requiredModelPath("PAPEROS_EMBEDDING_MODEL_PATH", "Embedding")
    : "";
  const rerankerEnabled = process.env.PAPEROS_RERANKER_ENABLED === "true";
  const rerankerModelPath = rerankerEnabled
    ? requiredModelPath("PAPEROS_RERANKER_MODEL_PATH", "Reranker")
    : "";
  return {
    host: process.env.PAPEROS_LOCAL_INFERENCE_HOST ?? "127.0.0.1",
    port: positiveInteger("PAPEROS_LOCAL_INFERENCE_PORT", 8081),
    embeddingEnabled,
    embeddingModelPath: modelPath,
    embeddingModelName: process.env.PAPEROS_EMBEDDING_MODEL_NAME ?? "embeddinggemma-300M",
    embeddingDimensions: positiveInteger("PAPEROS_EMBEDDING_DIMENSIONS", 768),
    embeddingMaxTokens: positiveInteger("PAPEROS_EMBEDDING_MAX_TOKENS", 2048),
    rerankerEnabled,
    rerankerModelPath,
    rerankerModelName: process.env.PAPEROS_RERANKER_MODEL_NAME ?? "qwen3-reranker-0.6b",
    rerankerMaxTokens: positiveInteger("PAPEROS_RERANKER_MAX_TOKENS", 4096),
    cudaVisibleDevices: process.env.CUDA_VISIBLE_DEVICES ?? "",
  };
}

function requiredModelPath(variable: string, label: string): string {
  const value = process.env[variable];
  if (!value) {
    throw new Error(`${variable} is required`);
  }
  const path = resolve(value);
  validateModel(label, path);
  return path;
}

function validateModel(label: string, path: string): void {
  accessSync(path, constants.R_OK);
  if (!statSync(path).isFile()) {
    throw new Error(`${label} model path is not a regular file: ${path}`);
  }
}
