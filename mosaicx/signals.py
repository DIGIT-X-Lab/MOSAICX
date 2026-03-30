# mosaicx/signals.py
"""Signals API client — proprietary radiology AI report evaluation.

Thin HTTP wrapper around the deepcOS Signals API.  Reports are
deidentified on-prem by MOSAICX before being sent for evaluation.

Usage::

    from mosaicx.signals import SignalsClient

    client = SignalsClient(api_key="sk_signals_...")
    result = client.evaluate(
        generated_report="AI report text...",
        reference_report="Reference report text...",
        metadata={"modality": "CT", "body_part": "chest"},
    )
    print(result["trust_score"], result["verdict"])
"""

from __future__ import annotations

from typing import Any

import requests

DEFAULT_BASE_URL = (
    "https://zyqsznjjxqmuyxeguqif.supabase.co/functions/v1/signals-api"
)

_TIMEOUT = 120  # seconds — evaluations can take a while


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SignalsError(Exception):
    """Base exception for all Signals API errors."""

    def __init__(self, message: str, code: str = "", status: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class SignalsAuthError(SignalsError):
    """Invalid or missing API key (HTTP 401)."""


class SignalsValidationError(SignalsError):
    """Invalid request parameters (HTTP 400)."""


class SignalsRateLimitError(SignalsError):
    """Rate limit exceeded (HTTP 429)."""

    def __init__(
        self,
        message: str,
        code: str = "rate_limit_exceeded",
        status: int = 429,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message, code, status)
        self.retry_after = retry_after


class SignalsUpstreamError(SignalsError):
    """Transient upstream failure (HTTP 502/504)."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SignalsClient:
    """HTTP client for the Signals evaluation API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = _TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    # -- Public API --------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """GET /health -- check service status."""
        return self._get("/health")

    def evaluate(
        self,
        generated_report: str,
        reference_report: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /evaluate -- evaluate AI report against reference.

        Parameters
        ----------
        generated_report : str
            The AI-generated report text (max 20,000 chars).
        reference_report : str
            The reference/ground-truth report text (max 20,000 chars).
        metadata : dict, optional
            Optional metadata: modality, body_part, model_name, model_version.

        Returns
        -------
        dict
            Full evaluation response including trust_score, verdict,
            metrics, safety_signals, structured_review, etc.
        """
        body: dict[str, Any] = {
            "mode": "report",
            "generated_report": generated_report,
            "reference_report": reference_report,
        }
        if metadata:
            body["metadata"] = metadata
        return self._post("/evaluate", body)

    def status(self, evaluation_id: str) -> dict[str, Any]:
        """GET /status/{evaluation_id} -- retrieve a past evaluation."""
        return self._get(f"/status/{evaluation_id}")

    # -- Internal ----------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        resp = self._session.get(
            f"{self._base_url}{path}", timeout=self._timeout
        )
        return self._handle_response(resp)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = self._session.post(
            f"{self._base_url}{path}", json=body, timeout=self._timeout
        )
        return self._handle_response(resp)

    def _handle_response(self, resp: requests.Response) -> dict[str, Any]:
        """Parse response, raising typed exceptions on errors."""
        try:
            data = resp.json()
        except ValueError as exc:
            raise SignalsError(
                f"Invalid response from Signals API (HTTP {resp.status_code})",
                code="invalid_response",
                status=resp.status_code,
            ) from exc

        if resp.ok:
            return data

        # Error response: {"error": {"code": "...", "message": "..."}}
        err = data.get("error", {})
        code = err.get("code", "unknown")
        message = err.get("message", resp.text)

        if resp.status_code == 401:
            raise SignalsAuthError(message, code=code, status=401)
        if resp.status_code == 400:
            raise SignalsValidationError(message, code=code, status=400)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            raise SignalsRateLimitError(
                message,
                code=code,
                retry_after=int(retry_after) if retry_after else None,
            )
        if resp.status_code in (502, 504):
            raise SignalsUpstreamError(message, code=code, status=resp.status_code)

        raise SignalsError(message, code=code, status=resp.status_code)
