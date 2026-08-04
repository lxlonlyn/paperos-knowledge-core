# Private Local Inference Protocol

- Verify enabled local files before binding the port.
- Never download models.
- Missing files return actionable errors.
- Endpoints: `/health`, `/v1/models`, `/v1/embeddings`, `/v1/rerank`, `/v1/query-expansion`.

- The PaperOS `Application` lifecycle is the sole owner of this private child
  process. It captures logs, waits for readiness, and terminates the process
  during server shutdown.
- Startup readiness is verified internally through `GET /health` before the
  PaperOS API begins accepting requests.
- A bound port is an actionable startup error; PaperOS never attaches to or
  terminates a gateway it did not start.
