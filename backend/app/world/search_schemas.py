from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Query parameters for search endpoint."""

    q: str = Field(..., description="Search query string", min_length=1)
    mode: Literal["literal", "semantic", "hybrid"] = Field(
        default="hybrid", description="Search mode"
    )
    k: int = Field(
        default=10, description="Maximum number of results to return", ge=1, le=100
    )
    types: Optional[str] = Field(
        default=None, description="Comma-separated list of memory types to filter by"
    )
    since: Optional[datetime] = Field(
        default=None, description="Only return memories updated after this timestamp"
    )


class ScoreExplanation(BaseModel):
    """Breakdown of how a search result was scored."""

    similarity: float = Field(..., description="Semantic similarity score [0,1]")
    literal_boost: float = Field(..., description="Literal substring match boost")
    recency_bonus: float = Field(..., description="Recency bonus [0,1]")
    type_bonus: float = Field(..., description="Type-specific bonus")


class SourceInfo(BaseModel):
    """Source information for a search result."""

    shard: str = Field(..., description="Shard name or 'session'")
    origin: str = Field(..., description="Origin type ('ingest' or 'memory')")


class SearchResult(BaseModel):
    """Individual search result."""

    item_id: str = Field(..., description="Unique identifier for the memory item")
    type: str = Field(..., description="Memory type (npc, location, etc.)")
    text: str = Field(..., description="Memory text content")
    score: float = Field(..., description="Total combined score")
    explanation: ScoreExplanation = Field(..., description="Score breakdown")
    updated_at: datetime = Field(..., description="When this memory was last updated")
    source: SourceInfo = Field(..., description="Source information")


class SearchResponse(BaseModel):
    """Response from search endpoint."""

    query: str = Field(..., description="The original search query")
    mode: str = Field(..., description="Search mode used")
    k: int = Field(..., description="Requested number of results")
    results: List[SearchResult] = Field(
        ..., description="Search results ordered by score"
    )


class ErrorResponse(BaseModel):
    """Error response with request correlation."""

    request_id: str = Field(..., description="Unique request identifier")
    message: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")
