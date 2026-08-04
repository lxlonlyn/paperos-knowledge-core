"""Private local inference client and child-process runtime."""

from paperos_core.runtime.local_inference.client import LocalInferenceClient
from paperos_core.runtime.local_inference.runtime import LocalInferenceRuntime

__all__ = ["LocalInferenceClient", "LocalInferenceRuntime"]
