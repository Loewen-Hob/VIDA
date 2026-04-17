# InteriorClarify Dataset Card

## Overview

**InteriorClarify** is a multimodal benchmark for proactive clarification in interior design. Each case represents an ambiguous user request and is annotated to support models that must decide:

1. what information is missing
2. which missing information matters most
3. how to ask a useful clarification question grounded in both text and images

According to the paper, the benchmark contains **1,016 real-world consultation cases**.

## Task Motivation

Many existing VLM systems assume that user intent is already complete. This assumption breaks in open-ended design scenarios, where users often provide partial goals, vague style descriptions, or incomplete spatial constraints. InteriorClarify is designed to evaluate whether a model can:

- identify missing design intents
- preserve visual grounding
- follow expert prioritization strategies
- ask actionable and context-aware follow-up questions


## Annotation Schema

The paper organizes design clarification around **nine critical design dimensions**:

- `Spatial Information`
- `Home Structure`
- `Budget`
- `Storage Requirements`
- `Target Users`
- `Style Preference`
- `Lifestyle Patterns`
- `Reference Images`
- `Personalized Needs`

Each target question is also associated with a **three-tier priority hierarchy**:

- `L3 Critical`
  Questions required before meaningful design can proceed.
- `L2 Influential`
  Questions that strongly affect design direction.
- `L1 Refinement`
  Questions that improve detail, precision, and personalization.

This hierarchy reflects how professional designers prioritize follow-up questions in real consultations.

## Data Collection

The paper reports that the benchmark is built from real consultation scenarios, with the following source distribution:

- `85%` online community platform cases
- `5%` online consultation cases
- `10%` offline consultation cases

The construction pipeline also includes image-supported data preparation and expert review. As illustrated in the paper, part of the data pipeline uses generated visual support during dataset construction.