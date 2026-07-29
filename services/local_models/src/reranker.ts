import {
  getLlama,
  type Llama,
  type LlamaModel,
  type LlamaRankingContext,
} from "node-llama-cpp";

import type {GatewayConfig} from "./config.js";

export interface RankedDocument {
  candidateId: string;
  originalIndex: number;
  relevanceScore: number;
  finalRank: number;
}

export class RerankerService {
  private llama: Llama | undefined;
  private model: LlamaModel | undefined;
  private context: LlamaRankingContext | undefined;

  public constructor(private readonly config: GatewayConfig) {}

  public async initialize(): Promise<void> {
    process.env.NODE_LLAMA_CPP_SKIP_DOWNLOAD = "true";
    this.llama = await getLlama({gpu: "auto"});
    this.model = await this.llama.loadModel({
      modelPath: this.config.rerankerModelPath,
    });
    this.context = await this.model.createRankingContext({
      contextSize: this.config.rerankerMaxTokens,
      batchSize: this.config.rerankerMaxTokens,
    });
  }

  public async rank(
    query: string,
    candidateIds: readonly string[],
    texts: readonly string[],
    limit: number,
  ): Promise<RankedDocument[]> {
    if (!this.context) {
      throw new Error("Reranker model is not initialized");
    }
    const scores = await this.context.rankAll(query, [...texts]);
    return scores
      .map((relevanceScore, originalIndex) => ({
        candidateId: candidateIds[originalIndex]!,
        originalIndex,
        relevanceScore,
        finalRank: 0,
      }))
      .sort((left, right) => right.relevanceScore - left.relevanceScore)
      .slice(0, limit)
      .map((item, index) => ({...item, finalRank: index + 1}));
  }

  public async dispose(): Promise<void> {
    if (this.context && !this.context.disposed) await this.context.dispose();
    if (this.model && !this.model.disposed) await this.model.dispose();
    if (this.llama && !this.llama.disposed) await this.llama.dispose();
  }
}
