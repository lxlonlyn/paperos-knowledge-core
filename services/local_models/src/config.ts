import {accessSync, constants, readFileSync, statSync} from "node:fs";
import {resolve} from "node:path";

export interface LocalInferenceConfig {
  host: string;
  port: number;
  embeddingModelPath: string;
  embeddingModelName: string;
  embeddingDimensions: number;
  embeddingMaxTokens: number;
  rerankerModelPath: string;
  rerankerModelName: string;
  rerankerMaxTokens: number;
  queryExpansionModelPath: string;
  queryExpansionModelName: string;
  queryExpansionMaxTokens: number;
  queryExpansionPrompt: string;
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
  const modelValue = process.env.PAPEROS_EMBEDDING_MODEL_PATH;
  if (!modelValue) {
    throw new Error("PAPEROS_EMBEDDING_MODEL_PATH is required");
  }
  const modelPath = resolve(modelValue);
  validateModel("Embedding", modelPath);
  const rerankerModelPath = requiredModelPath(
    "PAPEROS_RERANKER_MODEL_PATH",
    "Reranker",
  );
  const queryExpansionModelPath = requiredModelPath(
    "PAPEROS_QUERY_EXPANSION_MODEL_PATH",
    "Query expansion",
  );
  return {
    host: process.env.PAPEROS_LOCAL_INFERENCE_HOST ?? "127.0.0.1",
    port: positiveInteger("PAPEROS_LOCAL_INFERENCE_PORT", 8081),
    embeddingModelPath: modelPath,
    embeddingModelName: process.env.PAPEROS_EMBEDDING_MODEL_NAME ?? "embeddinggemma-300M",
    embeddingDimensions: positiveInteger("PAPEROS_EMBEDDING_DIMENSIONS", 768),
    embeddingMaxTokens: positiveInteger("PAPEROS_EMBEDDING_MAX_TOKENS", 2048),
    rerankerModelPath,
    rerankerModelName: process.env.PAPEROS_RERANKER_MODEL_NAME ?? "qwen3-reranker-0.6b",
    rerankerMaxTokens: positiveInteger("PAPEROS_RERANKER_MAX_TOKENS", 4096),
    queryExpansionModelPath,
    queryExpansionModelName:
      process.env.PAPEROS_QUERY_EXPANSION_MODEL_NAME ?? "qmd-query-expansion-1.7b",
    queryExpansionMaxTokens: positiveInteger(
      "PAPEROS_QUERY_EXPANSION_MAX_TOKENS",
      512,
    ),
    queryExpansionPrompt: requiredPrompt("PAPEROS_QUERY_EXPANSION_PROMPT_PATH"),
  };
}

function requiredPrompt(variable: string): string {
  const value = process.env[variable];
  if (!value) throw new Error(`${variable} is required`);
  const path = resolve(value);
  accessSync(path, constants.R_OK);
  const content = readFileSync(path, "utf8")
    .replace(/<!--\s*prompt-version:\s*[^\s]+\s*-->/u, "")
    .trim();
  if (!content) throw new Error(`Prompt file is empty: ${path}`);
  return content;
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
