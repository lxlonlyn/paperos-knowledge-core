import {createServer, type IncomingMessage, type ServerResponse} from "node:http";

import {loadConfig} from "./config.js";
import {EmbeddingService} from "./embedding.js";
import {errorPayload, RequestError} from "./errors.js";
import {QueryExpansionService} from "./queryExpansion.js";
import {RerankerService} from "./reranker.js";

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

interface QueryExpansionRequest {
  query?: string;
  profile?: string;
}

const config = loadConfig();
const embeddings = new EmbeddingService(config);
const reranker = new RerankerService(config);
const queryExpansion = new QueryExpansionService(config);
await embeddings.initialize();
await reranker.initialize();
await queryExpansion.initialize();

function send(response: ServerResponse, status: number, payload: object): void {
  response.writeHead(status, {"content-type": "application/json; charset=utf-8"});
  response.end(JSON.stringify(payload));
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
    if (request.method === "GET" && url.pathname === "/health") {
      send(response, 200, {
        status: "healthy",
        embedding: {
          model: config.embeddingModelName,
          dimensions: config.embeddingDimensions,
        },
        reranker: {model: config.rerankerModelName, loaded: true},
        query_expansion: {model: config.queryExpansionModelName, loaded: true},
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
          {
            id: config.rerankerModelName,
            object: "model",
            owned_by: "paperos-local",
            capabilities: ["rerank"],
          },
          {
            id: config.queryExpansionModelName,
            object: "model",
            owned_by: "paperos-local",
            capabilities: ["query-expansion"],
          },
        ],
      });
      return;
    }
    if (request.method === "POST" && url.pathname === "/v1/embeddings") {
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
      const results = await reranker.rank(
        body.query,
        body.candidate_ids,
        body.texts,
        limit,
      );
      send(response, 200, {
        model: config.rerankerModelName,
        results: results.map((item) => ({
          candidate_id: item.candidateId,
          original_index: item.originalIndex,
          relevance_score: item.relevanceScore,
          final_rank: item.finalRank,
        })),
      });
      return;
    }
    if (request.method === "POST" && url.pathname === "/v1/query-expansion") {
      const body = (await readJson(request)) as QueryExpansionRequest;
      if (typeof body.query !== "string" || body.query.trim().length === 0) {
        throw new RequestError("query must be a non-empty string");
      }
      const profile =
        typeof body.profile === "string" && body.profile.length > 0
          ? body.profile
          : "comprehensive";
      const expanded = await queryExpansion.expand(body.query, profile);
      send(response, 200, {
        model: config.queryExpansionModelName,
        lexical_queries: expanded.lexicalQueries,
        semantic_queries: expanded.semanticQueries,
        entity_queries: expanded.entityQueries,
        relation_queries: expanded.relationQueries,
        hyde_text: expanded.hydeText,
        raw_output: expanded.rawOutput,
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
      reranker_model: config.rerankerModelName,
      query_expansion_model: config.queryExpansionModelName,
    }) + "\n",
  );
});

let closing = false;
async function shutdown(signal: string): Promise<void> {
  if (closing) return;
  closing = true;
  process.stdout.write(JSON.stringify({event: "shutdown", signal}) + "\n");
  server.close();
  await queryExpansion.dispose();
  await reranker.dispose();
  await embeddings.dispose();
  process.exit(0);
}

process.on("SIGTERM", () => void shutdown("SIGTERM"));
process.on("SIGINT", () => void shutdown("SIGINT"));
