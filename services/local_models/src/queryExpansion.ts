import {
  getLlama,
  LlamaChatSession,
  type Llama,
  type LlamaModel,
} from "node-llama-cpp";

import type {GatewayConfig} from "./config.js";

export interface QueryExpansionResult {
  lexicalQueries: string[];
  semanticQueries: string[];
  entityQueries: string[];
  relationQueries: string[];
  hydeText: string;
  rawOutput: string;
}

export class QueryExpansionService {
  private llama: Llama | undefined;
  private model: LlamaModel | undefined;

  public constructor(private readonly config: GatewayConfig) {}

  public async initialize(): Promise<void> {
    process.env.NODE_LLAMA_CPP_SKIP_DOWNLOAD = "true";
    this.llama = await getLlama({gpu: "auto"});
    this.model = await this.llama.loadModel({
      modelPath: this.config.queryExpansionModelPath,
    });
  }

  public async expand(query: string, profile: string): Promise<QueryExpansionResult> {
    if (!this.model) {
      throw new Error("Query expansion model is not initialized");
    }
    let lastResult: QueryExpansionResult | undefined;
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      // A chat sequence retains KV state. Use a fresh context per attempt so
      // consecutive queries cannot inherit or collide with prior token state.
      const context = await this.model.createContext({
        contextSize: 4096,
        batchSize: 1024,
      });
      const session = new LlamaChatSession({
        contextSequence: context.getSequence(),
        systemPrompt:
          "Expand research queries for hybrid retrieval. Return concise JSON with " +
          "lexical_queries, semantic_queries, entity_queries, relation_queries, " +
          "and hyde_text. Preserve technical terms and do not answer from memory.",
      });
      try {
        const prompts = [
          `Profile: ${profile}\nOriginal query: ${query}`,
          "Task: produce multilingual hybrid-retrieval expansions as JSON.\n" +
            `Profile: ${profile}\nResearch question: ${JSON.stringify(query)}\n` +
            "Output one JSON object now:",
          "Rewrite this multilingual research question into several precise English " +
            "search queries, entities, and relations. Return the requested JSON only.\n" +
            `Question: ${query}\nJSON:`,
        ];
        const rawOutput = await session.prompt(
          prompts[attempt - 1]!,
          {
            maxTokens: this.config.queryExpansionMaxTokens,
            temperature: attempt === 1 ? 0.2 : 0.4,
          },
        );
        if (rawOutput.trim().length > 0) {
          lastResult = parseExpansion(query, rawOutput);
          if (hasGeneratedExpansion(lastResult, query) || attempt === 3) {
            return lastResult;
          }
        }
      } finally {
        session.dispose();
        await context.dispose();
      }
    }
    if (lastResult) return lastResult;
    throw new Error("Query expansion model returned empty output after 3 attempts");
  }

  public async dispose(): Promise<void> {
    if (this.model && !this.model.disposed) await this.model.dispose();
    if (this.llama && !this.llama.disposed) await this.llama.dispose();
  }
}

function parseExpansion(query: string, rawOutput: string): QueryExpansionResult {
  const cleaned = rawOutput.trim().replace(/^```(?:json)?\s*/u, "").replace(/```$/u, "");
  try {
    const parsed = JSON.parse(cleaned) as Record<string, unknown>;
    const hydeParams =
      parsed.hyde_params && typeof parsed.hyde_params === "object"
        ? (parsed.hyde_params as Record<string, unknown>)
        : {};
    return {
      lexicalQueries: stringList(
        parsed.lexical_queries ?? hydeParams.hyde_queries,
        query,
      ),
      semanticQueries: stringList(
        parsed.semantic_queries ?? hydeParams.hyde_queries,
        query,
      ),
      entityQueries: stringList(
        parsed.entity_queries ?? hydeParams.hyde_entities,
        query,
      ),
      relationQueries: stringList(
        parsed.relation_queries ?? hydeParams.hyde_relations,
        query,
      ),
      hydeText: typeof parsed.hyde_text === "string" ? parsed.hyde_text : cleaned,
      rawOutput,
    };
  } catch {
    const generated = cleaned
      .split(/\r?\n/u)
      .map((line) => line.replace(/^\s*(?:[-*]|\d+[.)])\s*/u, "").trim())
      .filter((line) => line.length > 3)
      .slice(0, 8);
    const queries = [...new Set([query, ...generated])];
    return {
      lexicalQueries: queries,
      semanticQueries: queries,
      entityQueries: queries,
      relationQueries: queries,
      hydeText: cleaned || query,
      rawOutput,
    };
  }
}

function stringList(value: unknown, original: string): string[] {
  if (typeof value === "string" && value.trim().length > 0) {
    return [...new Set([original, value.trim()])].slice(0, 8);
  }
  if (!Array.isArray(value)) return [original];
  const strings = value.filter(
    (item): item is string => typeof item === "string" && item.trim().length > 0,
  );
  return [...new Set([original, ...strings])].slice(0, 8);
}

function hasGeneratedExpansion(result: QueryExpansionResult, original: string): boolean {
  return [
    ...result.lexicalQueries,
    ...result.semanticQueries,
    ...result.entityQueries,
    ...result.relationQueries,
  ].some((value) => value.trim() !== original.trim());
}
