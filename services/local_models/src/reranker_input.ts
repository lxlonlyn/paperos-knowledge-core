export interface RerankerInputTokenTrace {
  queryTokenCount: number;
  documentTokenCount: number;
  specialPromptTokenCount: number;
  effectiveInputTokenCount: number;
  modelMaxInputTokens: number;
  truncated: false;
}

export class RerankerInputTooLargeError extends Error {
  public readonly code = "reranker_input_too_large";

  public constructor(
    public readonly effectiveInputTokenCount: number,
    public readonly modelMaxInputTokens: number,
  ) {
    super("The reranker input exceeds the configured model context.");
    this.name = "RerankerInputTooLargeError";
  }
}

/**
 * Validate token accounting for the exact model-template input.
 *
 * PaperOS currently rejects overflow instead of relying on implicit library or
 * model truncation. A successful trace is therefore truthfully non-truncated.
 */
export function validateRerankerInputTokenTrace(
  queryTokenCount: number,
  documentTokenCount: number,
  effectiveInputTokenCount: number,
  modelMaxInputTokens: number,
): RerankerInputTokenTrace {
  const specialPromptTokenCount =
    effectiveInputTokenCount - queryTokenCount - documentTokenCount;
  if (specialPromptTokenCount < 0) {
    throw new Error("Reranker token accounting is inconsistent.");
  }
  if (effectiveInputTokenCount > modelMaxInputTokens) {
    throw new RerankerInputTooLargeError(
      effectiveInputTokenCount,
      modelMaxInputTokens,
    );
  }
  return {
    queryTokenCount,
    documentTokenCount,
    specialPromptTokenCount,
    effectiveInputTokenCount,
    modelMaxInputTokens,
    truncated: false,
  };
}
