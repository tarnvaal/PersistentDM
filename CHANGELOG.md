# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Search API** (`GET /search`): Read-only memory search with hybrid ranking
  - Combines semantic similarity, literal substring matching, and recency boosting
  - Supports `literal`, `semantic`, and `hybrid` search modes
  - Configurable scoring weights via environment variables
  - Type filtering and timestamp-based filtering
  - Comprehensive per-result score explanations for debugging
  - Performance optimized for read-only access with snapshot-based queries
  - Full test coverage including unit tests, service tests, and E2E API tests

### Configuration
- Added search configuration module with environment variable support:
  - `SEARCH_W_SIM`, `SEARCH_W_LITERAL`, `SEARCH_W_REC`, `SEARCH_W_TYPE`
  - `SEARCH_HALF_LIFE_HOURS` for recency decay tuning
  - `SEARCH_TYPE_BONUS` for memory type-specific ranking adjustments

### Technical
- Added comprehensive test suite with deterministic fixtures
- Implemented performance benchmarks for scaling validation
- Added structured logging with request correlation
- Security validations for query length and timezone requirements
