import logging
import time
from typing import Optional, Literal
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from datetime import datetime

from ..world.search_service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


def get_request_id(request: Request) -> str:
    """Extract or generate request ID for correlation."""
    return getattr(request.state, "request_id", "unknown")


@router.get("")
async def search(
    request: Request,
    q: str = Query(
        ..., description="Search query string", min_length=1, max_length=512
    ),
    mode: Literal["literal", "semantic", "hybrid"] = Query(
        "hybrid", description="Search mode"
    ),
    k: int = Query(10, description="Maximum number of results", ge=1, le=100),
    types: Optional[str] = Query(
        None, description="Comma-separated list of memory types"
    ),
    since: Optional[datetime] = Query(
        None,
        description="Only return memories updated after this timestamp (ISO 8601 with timezone)",
    ),
    search_service: SearchService = Depends(SearchService),
):
    """Search memories with hybrid ranking."""
    request_id = get_request_id(request)
    start_time = time.time()

    try:
        # Validate mode (should be handled by Literal type, but double-check)
        if mode not in ["literal", "semantic", "hybrid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "request_id": request_id,
                    "message": f"Invalid mode: {mode}. Must be 'literal', 'semantic', or 'hybrid'",
                    "code": "INVALID_MODE",
                },
            )

        # Validate since parameter has timezone if provided
        if since and since.tzinfo is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "request_id": request_id,
                    "message": "since parameter must include timezone information (e.g., '2025-01-01T00:00:00Z' or '2025-01-01T00:00:00+00:00')",
                    "code": "INVALID_TIMEZONE",
                },
            )

        # Prepare filters
        filters = {}
        if types:
            filters["types"] = [t.strip() for t in types.split(",") if t.strip()]
        if since:
            filters["since"] = since

        # Perform search
        result = search_service.search(q=q, mode=mode, k=min(k, 100), filters=filters)

        # Log performance metrics
        elapsed_ms = (time.time() - start_time) * 1000
        pre_filter_count = len(search_service._gather_all_memories())
        post_filter_count = len(result["results"])

        logger.info(
            "Search completed",
            extra={
                "request_id": request_id,
                "mode": mode,
                "k": k,
                "pre_filter_count": pre_filter_count,
                "post_filter_count": post_filter_count,
                "timings_ms": {"total": round(elapsed_ms, 2)},
            },
        )

        return result

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(
            "Search failed",
            extra={
                "request_id": request_id,
                "error": str(e),
                "timings_ms": {"total": round(elapsed_ms, 2)},
            },
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "request_id": request_id,
                "message": "Internal server error during search",
                "code": "SEARCH_ERROR",
            },
        )
