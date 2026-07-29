export class RequestError extends Error {
  public constructor(
    message: string,
    public readonly statusCode = 400,
    public readonly code = "invalid_request",
  ) {
    super(message);
  }
}

export function errorPayload(error: unknown): object {
  if (error instanceof RequestError) {
    return {error: {message: error.message, type: error.code, code: error.code}};
  }
  const message = error instanceof Error ? error.message : String(error);
  return {
    error: {
      message,
      type: "model_gateway_error",
      code: "model_gateway_error",
    },
  };
}
