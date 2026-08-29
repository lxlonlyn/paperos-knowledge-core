import {
  getLlama,
  type Llama,
  type LlamaModel,
  type LlamaRankingContext,
} from "node-llama-cpp";

import type {LocalInferenceConfig} from "./config.js";
import {
  validateRerankerInputTokenTrace,
  type RerankerInputTokenTrace,
} from "./reranker_input.js";

export interface RankedDocument {
  candidateId: string;
  originalIndex: number;
  relevanceScore: number;
  finalRank: number;
  documentTokenCount: number;
  inputTokenCount: number;
  effectiveInputTokenCount: number;
  modelMaxInputTokens: number;
  queryTokenCount: number;
  specialPromptTokenCount: number;
  truncated: boolean;
  windowCount: number;
  winningWindowDocumentTokenCount: number;
  winningWindowIndex: number;
  winningWindowText: string;
}

// Every input is one prebuilt PaperOS RerankSpan. This service validates and
// scores spans without query-time sentence/window splitting. Python performs
// MaxP aggregation back to the authoritative canonical parent Chunk.

export class RerankerService {
  private llama: Llama | undefined;
  private model: LlamaModel | undefined;
  private context: LlamaRankingContext | undefined;

  public constructor(private readonly config: LocalInferenceConfig) {}

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
    if (!this.context || !this.model) {
      throw new Error("Reranker model is not initialized");
    }
    const queryTokenCount = this.model.tokenize(query).length;
    const documentTokenCounts = texts.map((text) => this.model!.tokenize(text).length);
    const inputTraces = texts.map((text, index) =>
      this.inputTokenTrace(query, queryTokenCount, text, documentTokenCounts[index]!),
    );
    const scores = await this.context.rankAll(query, Array.from(texts));
    const ranked = texts.map((text, originalIndex) => {
      const inputTrace = inputTraces[originalIndex]!;
      return {
        candidateId: candidateIds[originalIndex]!,
        originalIndex,
        relevanceScore: scores[originalIndex]!,
        finalRank: 0,
        documentTokenCount: documentTokenCounts[originalIndex]!,
        inputTokenCount: inputTrace.effectiveInputTokenCount,
        effectiveInputTokenCount: inputTrace.effectiveInputTokenCount,
        modelMaxInputTokens: inputTrace.modelMaxInputTokens,
        queryTokenCount: inputTrace.queryTokenCount,
        specialPromptTokenCount: inputTrace.specialPromptTokenCount,
        truncated: inputTrace.truncated,
        windowCount: 1,
        winningWindowDocumentTokenCount: documentTokenCounts[originalIndex]!,
        winningWindowIndex: 0,
        winningWindowText: text,
      };
    });
    return ranked
      .sort(
        (left, right) =>
          right.relevanceScore - left.relevanceScore || left.originalIndex - right.originalIndex,
      )
      .slice(0, limit)
      .map((item, index) => ({...item, finalRank: index + 1}));
  }

  private evaluationInputTokenCount(query: string, document: string): number {
    if (!this.context) return 0;
    const diagnosticContext = this.context as unknown as {
      _getEvaluationInput(queryText: string, documentText: string): readonly unknown[];
    };
    return diagnosticContext._getEvaluationInput(query, document).length;
  }

  private inputTokenTrace(
    query: string,
    queryTokenCount: number,
    text: string,
    documentTokenCount: number,
  ): RerankerInputTokenTrace {
    return validateRerankerInputTokenTrace(
      queryTokenCount,
      documentTokenCount,
      this.evaluationInputTokenCount(query, text),
      this.config.rerankerMaxTokens,
    );
  }

  public async dispose(): Promise<void> {
    if (this.context && !this.context.disposed) await this.context.dispose();
    if (this.model && !this.model.disposed) await this.model.dispose();
    if (this.llama && !this.llama.disposed) await this.llama.dispose();
  }
}
