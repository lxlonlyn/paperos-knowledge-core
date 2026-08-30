import {accessSync, constants, statSync} from "node:fs";
import {resolve} from "node:path";

export const LOCAL_INFERENCE_PROTOCOL_VERSION = 1;

export interface ModelFileIdentity {
  resolved_path: string;
  file_size: string;
  mtime_ns: string;
}

export interface RuntimeIdentity {
  protocol_version: number;
  embedding: {
    enabled: boolean;
    model: {
      name: string;
      file: ModelFileIdentity | null;
    };
    dimensions: number;
    max_tokens: number;
  };
  reranker: {
    enabled: boolean;
    model: {
      name: string;
      file: ModelFileIdentity | null;
    };
    max_tokens: number;
  };
  cuda_visible_devices: string;
}

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
  shutdownToken: string;
}

export function buildRuntimeIdentity(config: LocalInferenceConfig): RuntimeIdentity {
  return {
    protocol_version: LOCAL_INFERENCE_PROTOCOL_VERSION,
    embedding: {
      enabled: config.embeddingEnabled,
      model: {
        name: config.embeddingModelName,
        file: modelFileIdentity(config.embeddingEnabled, config.embeddingModelPath),
      },
      dimensions: config.embeddingDimensions,
      max_tokens: config.embeddingMaxTokens,
    },
    reranker: {
      enabled: config.rerankerEnabled,
      model: {
        name: config.rerankerModelName,
        file: modelFileIdentity(config.rerankerEnabled, config.rerankerModelPath),
      },
      max_tokens: config.rerankerMaxTokens,
    },
    cuda_visible_devices: config.cudaVisibleDevices,
  };
}

function modelFileIdentity(enabled: boolean, path: string): ModelFileIdentity | null {
  if (!enabled) return null;
  const metadata = statSync(path, {bigint: true});
  return {
    resolved_path: path,
    file_size: metadata.size.toString(),
    mtime_ns: metadata.mtimeNs.toString(),
  };
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
  const shutdownToken = process.env.PAPEROS_SHUTDOWN_TOKEN;
  if (!shutdownToken) {
    throw new Error("PAPEROS_SHUTDOWN_TOKEN is required");
  }
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
    shutdownToken,
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
