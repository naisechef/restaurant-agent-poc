# CLAUDE.md

# Restaurant Agent

Restaurant Agent is an evidence-first AI platform for discovering and ranking restaurants using structured data, web evidence, and LLM reasoning.

The initial capability focuses on identifying restaurants with outdoor seating, but the architecture should support additional place attributes without significant redesign.

---

# Project Goals

Build software that is:

- Reliable
- Modular
- Testable
- Explainable
- Easy to extend
- Production quality

Optimise for long-term maintainability rather than short-term speed.

---

# Preferred Technology Stack

## Language

- Python 3.12+

## Frameworks

- FastAPI
- Pydantic
- LangGraph

## LLM

- Anthropic Claude
- LLM providers should be abstracted behind interfaces.

## Data Processing

Prefer:

- Polars

Use Pandas only where ecosystem compatibility requires it.

## Testing

- pytest

## Code Quality

- Ruff
- MyPy

## Configuration

- Environment variables
- Pydantic Settings

---

# Architecture Principles

Follow a layered architecture.

```
API
 ↓
LangGraph Workflow
 ↓
Services
 ↓
Tools / Clients
 ↓
External Systems
```

Business logic belongs inside Services.

Avoid placing business logic inside:

- API routes
- prompts
- utility modules

---

# Agent Design

Prefer LangGraph whenever implementing:

- multi-step reasoning
- orchestration
- planning
- retrieval
- evidence collection
- human approval checkpoints

Avoid building custom orchestration unless there is a clear benefit.

Agent state should always be explicit and serialisable.

---

# Evidence First

Evidence is more important than conclusions.

Every extracted attribute should support:

- confidence
- evidence
- provenance
- source

Unknown is preferable to incorrect.

Never fabricate information.

---

# LLM Usage

LLMs should perform tasks such as:

- extraction
- summarisation
- reasoning
- classification

Business rules should remain in Python.

Keep prompts focused and reusable.

---

# External Services

Wrap external systems behind interfaces.

Never couple business logic directly to SDKs.

Prefer dependency injection where appropriate.

---

# Data Pipelines

Design data processing so components can evolve from:

- local execution
- batch processing
- distributed execution

Where large-scale processing becomes necessary, prefer patterns compatible with Spark rather than tightly coupling business logic to Spark itself.

---

# MCP Compatibility

Where appropriate:

- expose reusable capabilities as tools
- keep interfaces MCP-friendly
- avoid tightly coupling tools to specific agent implementations

---

# Coding Standards

Prefer:

- small functions
- explicit typing
- descriptive names
- composition over inheritance
- dependency injection
- deterministic behaviour

Avoid:

- global state
- hidden side effects
- duplicated logic
- large files
- unnecessary abstractions

---

# Testing

Every new feature should include appropriate tests.

Prefer:

- unit tests
- deterministic tests
- mocked external services

Avoid tests that require internet access.

---

# Security

Never commit:

- API keys
- credentials
- secrets

Validate all external input.

---

# Documentation

Keep documentation aligned with implementation.

When introducing significant architectural changes, update the relevant documents under `/docs`.

---

# Workflow

Before implementing significant changes:

1. Inspect the existing implementation.
2. Explain the proposed approach.
3. Identify affected files.
4. Keep changes small and reviewable.

After implementation:

1. Run available tests.
2. Run linting.
3. Run type checking.
4. Summarise the changes made.
5. Highlight assumptions and follow-up work.

---

# Decision Making

When multiple implementations are possible, prefer the solution that:

1. Improves maintainability.
2. Reduces coupling.
3. Increases testability.
4. Aligns with the existing architecture.
5. Keeps responsibilities clearly separated.