# Struct-XAI Platform

Struct-XAI is an experimental interpretability platform for studying how instruction-tuned language models form and revise decisions across layers.

The repository combines **LLM research code** with a lightweight **production-style service layer** so experiments can be submitted through an API, persisted in PostgreSQL, and exported to object storage.

## Focus

- layer-wise logit / decision-margin analysis
- structured ablation and span deletion
- candidate-aware attribution
- activation / hidden-state inspection
- cross-model comparisons
- experiment persistence and reproducibility
- API-first orchestration for interpretability jobs

## Architecture

```text
Client
  |
  v
FastAPI service
  |
  +--> PostgreSQL / RDS
  |      experiment metadata + metrics
  |
  +--> S3 / local artifact storage
  |      JSON outputs and reports
  |
  +--> Struct-XAI research runner
         layer analysis / attribution / ablation
```

The research code is deliberately kept inspectable: the platform is not intended to hide the underlying interpretability computations behind an opaque service.

## Research core

`research/struct_xai_core.py` contains a compact layer-wise analysis example using hidden states and the model language-model head. It demonstrates the key idea behind the project: inspect how token-level decisions change across layers, then perturb parts of the input and compare the resulting internal trajectories.

## Service layer

The `cloud_service/` package wraps experiments with:

- FastAPI endpoints
- experiment IDs and status
- PostgreSQL persistence
- artifact storage
- container-ready execution

This turns one-off interpretability notebooks/scripts into trackable experiments.

## Public-project boundary

This repository contains public research/demo code only. It does not contain employer data, proprietary prompts, internal model artifacts, or confidential evaluation datasets.

## Roadmap

- migrate the remaining Struct-XAI research scripts from the original working repository
- add reproducible small-model examples
- add token-layer heatmap output
- add activation-patching experiments
- add experiment queue / background worker
- add Docker Compose and Terraform reference deployment
- add CI smoke tests

## Portfolio context

This project demonstrates both sides of applied LLM work: **model-level interpretability research** and the **software systems required to operationalize experiments**.
