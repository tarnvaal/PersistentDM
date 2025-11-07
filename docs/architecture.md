# PersistentDM Architecture Overview

## System Architecture

PersistentDM is a FastAPI-based AI Dungeon Master system with a layered architecture designed for conversational world-building and memory management.

## Core Components

### Application Layer
- **FastAPI Framework**: REST API with async endpoints, dependency injection, and middleware
- **CORS Configuration**: Flexible frontend integration with dynamic port detection
- **Health Checks**: Readiness/liveness probes for container orchestration

### Service Layer
- **ConversationService**: Orchestrates chat interactions with context building and memory analysis
- **StateService**: Manages world state lifecycle (reset, persistence, monitoring)
- **Session Service**: Handles save/load/merge of conversation states and world data

### Domain Layer (world/)
- **WorldMemory**: Central memory management with embedding-based retrieval
- **Context Builder**: Multi-index memory retrieval with scoring and formatting
- **Location Graph**: Spatial navigation and relationship modeling
- **NPC Index**: Character relationship and state tracking

### Infrastructure Layer (utility/)
- **Chatter (LLM)**: Singleton Llama.cpp wrapper with shared model loading
- **Embeddings**: SentenceTransformer singleton for semantic similarity
- **History**: Token-aware conversation management
- **GPU Utilities**: VRAM monitoring and resource management

## Key Architectural Patterns

### Singleton Pattern
- Expensive resources (LLM, embeddings) use thread-safe singletons
- Prevents multiple model loads and ensures consistent instances

### Dependency Injection
- FastAPI DI system with @lru_cache for service instantiation
- Clean separation between route handlers and business logic

### Layered Architecture
- Clear separation: API → Service → Domain → Infrastructure
- Each layer has distinct responsibilities and interfaces

### Fail-Safe Design
- Graceful degradation when LLM features unavailable
- Best-effort persistence with silent error handling
- Defensive programming throughout (type checking, exception handling)

### Streaming & Async
- Server-Sent Events for real-time chat and ingest progress
- Background task coordination for model loading

## Data Flow Patterns

### Chat Flow
User Message → Context Retrieval → LLM Processing → Memory Analysis → State Update → Response

### Ingest Flow
Text Upload → Streaming Chunking → LLM Memory Extraction → Embedding → Persistence

### Search Flow (Proposed)
Query → Embedding → Multi-index Retrieval → Result Ranking → Response

## State Management

### In-Memory State
- Session memories, NPC index, location graph
- Ingest layers (persistent shards loaded into memory)
- Thread-safe with RLock protection

### Persistence
- JSON-based session storage with metadata
- Ingest shards persisted to filesystem
- Vector recomputation on load (don't trust persisted embeddings)

## Design Decisions

### Memory Architecture
- Dual embeddings per memory (explanation + window context)
- Multi-index retrieval (general, entity-specific, type-specific)
- Recency + similarity + type bonus scoring
- Confidence thresholds for LLM operations

### LLM Integration
- Signature inspection for optional features
- Fallback behavior when capabilities unavailable
- Token-aware history management
- Context injection via world_facts parameter

### Error Handling
- Structured logging with request correlation
- Debug mode error exposure via environment variables
- HTTP status code consistency
- Graceful service degradation

## Proposed Extensions

### Search Feature
- Add to world/ module as read-only retrieval service
- Leverage existing embedding and memory infrastructure
- Support literal text, semantic, and hybrid search modes

### Assistant Mode
- Tool orchestration layer above existing services
- Intent analysis and tool selection
- API-based integration with existing chat/ingest/search endpoints

## Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **LLM**: llama.cpp with GGUF models (LatitudeGames/Harbinger-24B-GGUF)[Q5_K_M]
- **Embeddings**: SentenceTransformers (BAAI/bge-small-en-v1.5)
- **Persistence**: JSON files (sessions, ingest shards)
- **Testing**: pytest with FastAPI TestClient
- **Logging**: Structured JSON logging with context correlation
