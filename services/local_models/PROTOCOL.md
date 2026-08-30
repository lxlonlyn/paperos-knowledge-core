# Private Local Inference Protocol

- Verify enabled local files before binding the port.
- Never download models.
- Missing files return actionable errors.
- Business endpoints: `/health`, `/v1/models`, `/v1/embeddings`, `/v1/rerank`.
- `POST /internal/shutdown` is reserved for the Python parent and requires the
  random token supplied in `PAPEROS_SHUTDOWN_TOKEN` at child creation.

- The PaperOS `Application` lifecycle is the sole owner of this private child
  process. It captures logs, waits for readiness, and uses the authenticated
  shutdown endpoint during normal server shutdown. Process termination and
  killing are timeout/error fallbacks only.
- Startup readiness is verified internally through `GET /health` before the
  PaperOS API begins accepting requests.
- A healthy runtime left across a PaperOS restart may be reused only when the
  `runtime_identity` reported by `GET /health` exactly matches the identity
  expected from the current protocol, model configuration, lightweight model
  file metadata, and CUDA visibility.
- An occupied endpoint with an absent or mismatched identity is an actionable
  compatibility error. PaperOS does not shut down or terminate a process it did
  not start.
