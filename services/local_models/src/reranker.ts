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

interface RankingWindow {
  index: number;
  text: string;
  documentTokenCount: number;
}

// Current reranker windowing is a temporary query-time projection. These
// parameters and max-score aggregation are provisional; they do not define
// the final PaperOS reranking architecture. The authoritative indexed and evidence unit
// remains the parent canonical Chunk.
const PROVISIONAL_WINDOW_DOCUMENT_TOKENS = 96;
const PROVISIONAL_WINDOW_TOKEN_OVERLAP = 16;
const SCALAR_RETRIEVAL_HEADER = /^(?:Paper|Year|Region|Section):\n/u;
const SENTENCE_END = new Set([".", "!", "?", "。", "！", "？"]);
const NON_TERMINAL_ABBREVIATIONS = new Set([
  "al",
  "dr",
  "eq",
  "eqs",
  "e.g",
  "fig",
  "figs",
  "i.e",
  "mr",
  "mrs",
  "prof",
  "sec",
  "secs",
  "vs",
]);

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
    const windows = texts.map((text) => this.windows(text));
    const flatWindows = windows.flat();
    const queryTokenCount = this.model.tokenize(query).length;
    const inputTraces = flatWindows.map((window) =>
      this.inputTokenTrace(query, queryTokenCount, window),
    );
    const scores = await this.context.rankAll(
      query,
      flatWindows.map((window) => window.text),
    );
    let scoreIndex = 0;
    let traceIndex = 0;
    const ranked = windows.map((documentWindows, originalIndex) => {
      const documentScores = scores.slice(scoreIndex, scoreIndex + documentWindows.length);
      const documentInputTraces = inputTraces.slice(
        traceIndex,
        traceIndex + documentWindows.length,
      );
      scoreIndex += documentWindows.length;
      traceIndex += documentWindows.length;
      let winningWindowIndex = 0;
      for (let index = 1; index < documentScores.length; index += 1) {
        if (documentScores[index]! > documentScores[winningWindowIndex]!) {
          winningWindowIndex = index;
        }
      }
      const winningWindow = documentWindows[winningWindowIndex]!;
      const winningInputTrace = documentInputTraces[winningWindowIndex]!;
      const relevanceScore = documentScores[winningWindowIndex]!;
      return {
        candidateId: candidateIds[originalIndex]!,
        originalIndex,
        relevanceScore,
        finalRank: 0,
        documentTokenCount: this.model!.tokenize(texts[originalIndex]!).length,
        inputTokenCount: winningInputTrace.effectiveInputTokenCount,
        effectiveInputTokenCount: winningInputTrace.effectiveInputTokenCount,
        modelMaxInputTokens: winningInputTrace.modelMaxInputTokens,
        queryTokenCount: winningInputTrace.queryTokenCount,
        specialPromptTokenCount: winningInputTrace.specialPromptTokenCount,
        truncated: winningInputTrace.truncated,
        windowCount: documentWindows.length,
        winningWindowDocumentTokenCount: winningWindow.documentTokenCount,
        winningWindowIndex,
        winningWindowText: winningWindow.text,
      };
    });
    return ranked
      .sort((left, right) => right.relevanceScore - left.relevanceScore)
      .slice(0, limit)
      .map((item, index) => ({...item, finalRank: index + 1}));
  }

  private windows(text: string): RankingWindow[] {
    if (!this.model) {
      throw new Error("Reranker model is not initialized");
    }
    const blocks = text.split(/\n\n+/u).filter((block) => block.length > 0);
    const headerBlocks: string[] = [];
    while (blocks.length > 0 && SCALAR_RETRIEVAL_HEADER.test(blocks[0]!)) {
      headerBlocks.push(blocks.shift()!);
    }
    const prefix = headerBlocks.join("\n\n");
    const bodyBlocks = blocks.length > 0 ? blocks : [text];
    const windowTexts = bodyBlocks.flatMap((block) => this.windowsForBlock(prefix, block));
    return windowTexts.map((windowText, index) => ({
      index,
      text: windowText,
      documentTokenCount: this.model!.tokenize(windowText).length,
    }));
  }

  private windowsForBlock(prefix: string, block: string): string[] {
    const render = (body: string): string => (prefix ? `${prefix}\n\n${body}` : body);
    if (this.documentTokenCount(render(block)) <= PROVISIONAL_WINDOW_DOCUMENT_TOKENS) {
      return [render(block)];
    }
    const sentences = splitSentences(block);
    const windows: string[] = [];
    let start = 0;
    while (start < sentences.length) {
      let end = start;
      while (
        end < sentences.length &&
        this.documentTokenCount(render(sentences.slice(start, end + 1).join(" "))) <=
          PROVISIONAL_WINDOW_DOCUMENT_TOKENS
      ) {
        end += 1;
      }
      if (end === start) {
        windows.push(...this.tokenWindows(prefix, sentences[start]!));
        start += 1;
        continue;
      }
      windows.push(render(sentences.slice(start, end).join(" ")));
      if (end >= sentences.length) break;
      start = end - start > 1 ? end - 1 : end;
    }
    return windows;
  }

  private tokenWindows(prefix: string, text: string): string[] {
    if (!this.model) return [];
    const render = (body: string): string => (prefix ? `${prefix}\n\n${body}` : body);
    const prefixTokens = this.documentTokenCount(render(""));
    const available = Math.max(1, PROVISIONAL_WINDOW_DOCUMENT_TOKENS - prefixTokens);
    const tokens = this.model.tokenize(text);
    const windows: string[] = [];
    let start = 0;
    while (start < tokens.length) {
      const end = Math.min(tokens.length, start + available);
      windows.push(render(this.model.detokenize(tokens.slice(start, end))));
      if (end >= tokens.length) break;
      start = Math.max(start + 1, end - PROVISIONAL_WINDOW_TOKEN_OVERLAP);
    }
    return windows;
  }

  private documentTokenCount(text: string): number {
    if (!this.model) return 0;
    return this.model.tokenize(text).length;
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
    window: RankingWindow,
  ): RerankerInputTokenTrace {
    return validateRerankerInputTokenTrace(
      queryTokenCount,
      window.documentTokenCount,
      this.evaluationInputTokenCount(query, window.text),
      this.config.rerankerMaxTokens,
    );
  }

  public async dispose(): Promise<void> {
    if (this.context && !this.context.disposed) await this.context.dispose();
    if (this.model && !this.model.disposed) await this.model.dispose();
    if (this.llama && !this.llama.disposed) await this.llama.dispose();
  }
}

function splitSentences(text: string): string[] {
  const sentences: string[] = [];
  let start = 0;
  for (let index = 0; index < text.length; index += 1) {
    if (!SENTENCE_END.has(text[index]!)) continue;
    let next = index + 1;
    while (next < text.length && /\s/u.test(text[next]!)) next += 1;
    if (next >= text.length) {
      sentences.push(text.slice(start).trim());
      start = text.length;
      break;
    }
    if (!startsSentence(text[next]!) || isNonTerminalAbbreviation(text, index)) continue;
    sentences.push(text.slice(start, index + 1).trim());
    start = next;
    index = next - 1;
  }
  if (start < text.length) sentences.push(text.slice(start).trim());
  return sentences.filter((sentence) => sentence.length > 0);
}

function startsSentence(character: string): boolean {
  return /[A-ZÀ-ÖØ-Þ\u3400-\u9fff"“‘([]/u.test(character);
}

function isNonTerminalAbbreviation(text: string, periodIndex: number): boolean {
  if (text[periodIndex] !== ".") return false;
  const prefix = text.slice(0, periodIndex).toLowerCase();
  const match = prefix.match(/([a-z](?:[a-z.]*)?)$/u);
  return match !== null && NON_TERMINAL_ABBREVIATIONS.has(match[1]!);
}
