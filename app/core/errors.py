"""Application error hierarchy and FastAPI exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base class for domain errors mapped to HTTP responses."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class UnauthorizedError(AppError):
    """Missing or invalid authentication (no/invalid bearer token)."""

    status_code = 401
    code = "unauthorized"


class ForbiddenError(AppError):
    """Authenticated, but not permitted (catalogue authorization denied)."""

    status_code = 403
    code = "forbidden"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class GoneError(AppError):
    """The retained resource exists, but its physical media has expired."""

    status_code = 410
    code = "asset_media_expired"


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"


class BadRequestError(AppError):
    status_code = 400
    code = "bad_request"


class CapabilityUnavailableError(ValidationError):
    """Safe readiness rejection at admission or during queued execution.

    The message is a plain sentence for end users; the machine-readable reason
    travels separately as reason_code in the error body.
    """

    code = "capability_unavailable"

    def __init__(self, message: str, *, reason_code: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class PreconditionFailedError(AppError):
    status_code = 412
    code = "precondition_failed"


class PreconditionRequiredError(AppError):
    status_code = 428
    code = "precondition_required"


class UnsupportedMediaTypeError(AppError):
    status_code = 415
    code = "unsupported_media_type"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class ProviderError(AppError):
    """Raised when a hosted-inference provider call fails."""

    status_code = 502
    code = "provider_error"


class ToolError(AppError):
    """Raised when a local CLI tool (ffmpeg/imagemagick) fails."""

    status_code = 500
    code = "tool_error"


def _error_body(code: str, message: str, reason_code: str | None = None) -> dict:
    error: dict[str, str] = {"code": code, "message": message}
    if reason_code is not None:
        error["reason_code"] = reason_code
    return {"error": error}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message, getattr(exc, "reason_code", None)),
        )
