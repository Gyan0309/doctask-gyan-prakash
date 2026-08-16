"""Outbound integrations with systems we do not own.

Kept out of `domain/` and `services/` deliberately: everything in here can fail because
someone else's service is having a bad day, and none of it may be able to fail a run.
"""

from integrations.superdocs import (
    SuperDocsClient,
    SuperDocsError,
    SuperDocsUnavailable,
)

__all__ = ["SuperDocsClient", "SuperDocsError", "SuperDocsUnavailable"]
