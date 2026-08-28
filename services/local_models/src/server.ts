import {timingSafeEqual} from "node:crypto";
import {createServer, type IncomingMessage, type ServerResponse} from "node:http";

import {loadConfig} from "./config.js";
import {EmbeddingService} from "./embedding.js";
import {errorPayload, RequestError} from "./errors.js";
import {RerankerService} from "./reranker.js";
import {RerankerInputTooLargeError} from "./reranker_input.js";

interface EmbeddingRequest {
  input: string | string[];
  model?: string;
}

interface RerankRequest {
  query?: string;
  candidate_ids?: string[];
  texts?: string[];
  limit?: number;
}

const config = loadConfig();
const embeddings = config.embeddingEnabled ? new EmbeddingService(config) : null;
const reranker = new RerankerService(config);
if (embeddings) {
  await embeddings.initialize();
}
if (config.rerankerEnabled) {
  await reranker.initialize();
}

function send(response: ServerResponse, status: number, payload: object): void {
  response.writeHead(status, {"content-type": "application/json; charset=utf-8"});
  response.end(JSON.stringify(payload));
}

function validShutdownToken(value: string | string[] | undefined): boolean {
  if (typeof value !== "string") return false;
  const supplied = Buffer.from(value);
  const expected = Buffer.from(config.shutdownToken);
  return supplied.length === expected.length && timingSafeEqual(supplied, expected);
}

async function readJson(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buffer.length;
    if (size > 8 * 1024 * 1024) {
      throw new RequestError("Request body exceeds 8 MiB", 413, "request_too_large");
    }
    chunks.push(buffer);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new RequestError("Request body must be valid JSON");
  }
}

function embeddingInput(value: unknown): string[] {
  if (typeof value === "string" && value.length > 0) {
    return [value];
  }
  if (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((item) => typeof item === "string" && item.length > 0)
  ) {
    return value;
  }
  throw new RequestError("input must be a non-empty string or string array");
}

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url ?? "/", `http://${config.host}:${config.port}`);
    if (request.method === "POST" && url.pathname === "/internal/shutdown") {
      if (!validShutdownToken(request.headers["x-paperos-shutdown-token"])) {
        throw new RequestError("Invalid shutdown token", 403, "forbidden");
      }
      send(response, 202, {status: "shutting_down"});
      setImmediate(() => void shutdown("parent"));
      return;
    }
    if (request.method === "GET" && url.pathname === "/health") {
      send(response, 200, {
        status: "healthy",
        cuda_visible_devices: config.cudaVisibleDevices,
        embedding: {
          model: config.embeddingModelName,
          dimensions: config.embeddingDimensions,
          loaded: config.embeddingEnabled,
        },
        reranker: {
          model: config.rerankerModelName,
          loaded: config.rerankerEnabled,
        },
      });
      return;
    }
    if (request.method === "GET" && url.pathname === "/v1/models") {
      send(response, 200, {
        object: "list",
        data: [
          {
            id: config.embeddingModelName,
            object: "model",
            owned_by: "paperos-local",
            capabilities: ["embeddings"],
          },
          ...(config.rerankerEnabled
            ? [
                {
                  id: config.rerankerModelName,
                  object: "model",
                  owned_by: "paperos-local",
                  capabilities: ["rerank"],
                },
              ]
            : []),
        ],
      });
      return;
    }
    if (request.method === "POST" && url.pathname === "/v1/embeddings") {
      if (!embeddings) {
        throw new RequestError("Embedding is disabled for this runtime", 404, "embedding_disabled");
      }
      const body = (await readJson(request)) as Partial<EmbeddingRequest>;
      const inputs = embeddingInput(body.input);
      const vectors = await embeddings.embed(inputs);
      send(response, 200, {
        object: "list",
        model: body.model ?? config.embeddingModelName,
        data: vectors.map((embedding, index) => ({
          object: "embedding",
          index,
          embedding,
        })),
        usage: {
          prompt_tokens: inputs.reduce((total, input) => total + Math.ceil(input.length / 4), 0),
          total_tokens: inputs.reduce((total, input) => total + Math.ceil(input.length / 4), 0),
        },
      });
      return;
    }
    if (request.method === "POST" && url.pathname === "/v1/rerank") {
      if (!config.rerankerEnabled) {
        throw new RequestError(
          "Reranker is disabled; enable retrieval.rerank_enabled",
          404,
          "reranker_disabled",
        );
      }
      const body = (await readJson(request)) as RerankRequest;
      if (
        typeof body.query !== "string" ||
        !Array.isArray(body.candidate_ids) ||
        !Array.isArray(body.texts) ||
        body.candidate_ids.length === 0 ||
        body.candidate_ids.length !== body.texts.length ||
        !body.candidate_ids.every((item) => typeof item === "string" && item.length > 0) ||
        !body.texts.every((item) => typeof item === "string" && item.length > 0)
      ) {
        throw new RequestError(
          "query, equally-sized candidate_ids and texts are required",
        );
      }
      const limit =
        Number.isSafeInteger(body.limit) && (body.limit ?? 0) > 0
          ? Math.min(body.limit!, body.texts.length)
          : body.texts.length;
      let results;
      try {
        results = await reranker.rank(
          body.query,
          body.candidate_ids,
          body.texts,
          limit,
        );
      } catch (error) {
        if (error instanceof RerankerInputTooLargeError) {
          throw new RequestError(error.message, 422, error.code);
        }
        throw error;
      }
      send(response, 200, {
        model: config.rerankerModelName,
        results: results.map((item) => ({
          candidate_id: item.candidateId,
          original_index: item.originalIndex,
          relevance_score: item.relevanceScore,
          final_rank: item.finalRank,
          document_token_count: item.documentTokenCount,
          input_token_count: item.inputTokenCount,
          effective_input_token_count: item.effectiveInputTokenCount,
          model_max_input_tokens: item.modelMaxInputTokens,
          query_token_count: item.queryTokenCount,
          special_prompt_token_count: item.specialPromptTokenCount,
          truncated: item.truncated,
          window_count: item.windowCount,
          winning_window_document_token_count: item.winningWindowDocumentTokenCount,
          winning_window_index: item.winningWindowIndex,
          winning_window_text: item.winningWindowText,
        })),
      });
      return;
    }
    throw new RequestError("Endpoint not found", 404, "not_found");
  } catch (error) {
    const status = error instanceof RequestError ? error.statusCode : 500;
    process.stderr.write(
      JSON.stringify({
        event: "request_error",
        status,
        message: error instanceof Error ? error.message : String(error),
      }) + "\n",
    );
    send(response, status, errorPayload(error));
  }
});

server.listen(config.port, config.host, () => {
  process.stdout.write(
    JSON.stringify({
      event: "ready",
      host: config.host,
      port: config.port,
      embedding_model: config.embeddingModelName,
      reranker_model: config.rerankerEnabled
        ? config.rerankerModelName
        : "disabled",
      cuda_visible_devices: config.cudaVisibleDevices,
    }) + "\n",
  );
});

let closing = false;
async function shutdown(signal: string): Promise<void> {
  if (closing) return;
  closing = true;
  process.stdout.write(JSON.stringify({event: "shutdown", signal}) + "\n");
  await new Promise<void>((resolve) => server.close(() => resolve()));
  await reranker.dispose();
  if (embeddings) {
    await embeddings.dispose();
  }
  process.exit(0);
}

process.on("SIGTERM", () => void shutdown("SIGTERM"));
process.on("SIGINT", () => void shutdown("SIGINT"));
