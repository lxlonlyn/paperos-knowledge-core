import {getLlama, type Llama, type LlamaEmbeddingContext, type LlamaModel} from "node-llama-cpp";

import type {GatewayConfig} from "./config.js";

export class EmbeddingService {
  private llama: Llama | undefined;
  private model: LlamaModel | undefined;
  private context: LlamaEmbeddingContext | undefined;

  public constructor(private readonly config: GatewayConfig) {}

  public async initialize(): Promise<void> {
    process.env.NODE_LLAMA_CPP_SKIP_DOWNLOAD = "true";
    this.llama = await getLlama({gpu: "auto"});
    this.model = await this.llama.loadModel({
      modelPath: this.config.embeddingModelPath,
    });
    if (this.model.embeddingVectorSize !== this.config.embeddingDimensions) {
      throw new Error(
        `Embedding dimension mismatch: model=${this.model.embeddingVectorSize}, ` +
          `configured=${this.config.embeddingDimensions}`,
      );
    }
    this.context = await this.model.createEmbeddingContext({
      contextSize: this.config.embeddingMaxTokens,
      batchSize: this.config.embeddingMaxTokens,
    });
  }

  public async embed(inputs: readonly string[]): Promise<readonly number[][]> {
    if (!this.context || !this.model) {
      throw new Error("Embedding model is not initialized");
    }
    const vectors: number[][] = [];
    for (const input of inputs) {
      const tokens = this.model.tokenize(input);
      const limited =
        tokens.length <= this.config.embeddingMaxTokens
          ? input
          : this.model.detokenize(tokens.slice(0, this.config.embeddingMaxTokens));
      const embedding = await this.context.getEmbeddingFor(limited);
      vectors.push(Array.from(embedding.vector));
    }
    return vectors;
  }

  public async dispose(): Promise<void> {
    if (this.context && !this.context.disposed) {
      await this.context.dispose();
    }
    if (this.model && !this.model.disposed) {
      await this.model.dispose();
    }
    if (this.llama && !this.llama.disposed) {
      await this.llama.dispose();
    }
  }
}
