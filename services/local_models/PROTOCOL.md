# Local Model Gateway Protocol

- Verify enabled local files before binding the port.
- Never download models.
- Missing files return actionable errors.
- Endpoints: `/health`, `/v1/models`, `/v1/embeddings`, `/v1/rerank`, `/v1/query-expansion`.

- The formal `paperos model-gateway` command owns this server process, inherits
  its stdout/stderr, and remains in the foreground until SIGINT or SIGTERM.
- Startup readiness is announced as
  `Model gateway listening on http://<host>:<port>`.
- A bound port is an actionable startup error; the command never attaches to or
  terminates a gateway it did not start.
