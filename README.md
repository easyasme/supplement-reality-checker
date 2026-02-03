# Supplement Claim Reality Checker
### Multi-Agent RAG System with local LLMs and A/B Prompt Evaluation

This project implements a **multi-agent retrieval-augmented generation (RAG) system** for verifying dietary supplement claims using **local large language models**.

Instead of generating medical advice, the system evaluates whether a supplement claim is **Supported**, **Not Supported**, **Mixed**, or **Unknown**, based strictly on **public, authoritative evidence** such as documents from the NIH Office of Dietary Supplements.

The system is designed for **claim verification and safety-oriented reasoning**, not for conversational or generative health assistance.

---

## Problem Overview

Dietary supplement marketing often contains claims that are:

- Vague or loosely defined
- Overstated compared to available evidence
- Difficult for consumers to verify on their own

At the same time, directly using LLMs for health-related questions introduces serious risks:

- Hallucinated medical explanations
- Unsupported causal conclusions
- Overconfident or unsafe recommendations

This project addresses these risks by enforcing the following constraints:

- **Evidence-first reasoning** (no external or prior knowledge)
- **Mandatory citation of retrieved sources**
- **Guardrails against diagnosis, treatment, or dosage advice**

---

## System Design

The system is implemented as a sequence of explicit steps, where each stage has a clearly defined responsibility.

### Pipeline Overview

Input (text or image)

-> OCR (Hugging Face, optional)

-> Retrieval (FAISS)

-> Evidence Judge (LLM, evidence-only)

-> Verdict + Citations + Guardrails

-> Telemetry logging

---

## Data Sources

- NIH Office of Dietary Supplements
- Official public health fact sheets and documentation

Documents are processed as follows:

- Downloaded as HTML or PDF
- Converted into plain text
- Chunked into semantically meaningful sections
- Embedded using sentence-level transformer models

---

### Vision-based Input (OCR)

In addition to text-based claims, the system supports image-based input such as supplement label photos.

A lightweight OCR agent is used to extract raw label text from images before passing it into the RAG pipelien.

- OCR model: Hugging Face TrOCR (`microsoft/trocr-small-printed`)
- Input: supplement label image
- Output: extracted plain text

This design allows the system to handle real-world supplement packaging, where claims are often presented visually rather than as clean text.

---

## Retrieval Agent

- Vector database: FAISS (cosine similarity)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Top-k retrieval (default: k = 5)

Each retrieved chunk retains:

- Document ID
- Section or chunk ID
- original source attribution

The retrieval agent is intentionally kept simple to ensure interpretability.

---

## Evidence Judge Agent (LLM)

The judge agent is responsible only for evaluating the retrieved evidence.

It operates under strict rules:

- Uses **only the retrieved evidence**
- If evidence is insufficient, outputs `Unknown`
- No diagnosis, treatment, or dosage recommendations
- Outputs **structured JSON only**

Inference runs locally using **Ollama (llama3.1:8b)**.

### Output Schema

```json
{
  "verdict": "Supported | Mixed | NotSupported | Unknown",
  "short_reason": "1–2 evidence-based sentences",
  "citations": ["chunk_id_1", "chunk_id_2"],
  "confidence": 0.0 – 1.0
}
```

---

## Example Results

### Example - Supported Claim

**Input**
Omega-3 supplements can lower triglyceride levels.

**Judge Output**

```json
{
  "verdict": "Supported",
  "short_reason": "Omega-3 supplements lower triglyceride levels by about 15% and reduce the risk of cardiac death.",
  "citations": [
    "nih_ods_omega3::chunk_29",
    "nih_ods_omega3::chunk_30"
  ],
  "confidence": 0.8
}
```

---

## Telemetry Logging

Every system run is logged to a local SQLite database, including:

- Raw user input
- Retrieved evidence chunks
- Judge output JSON
- Guardrail flags
- Timestamp

This enables:

- Offline failure analysis
- Prompt iteration without rerunning inference
- Reproducible evaluation

---

## Evaluation

A dedicated evaluation runner executes batch tests on a predefined failure set.

### Metrics Tracked

- Verdict accuracy
- Citation hit rate
- Guardrail activation rate

### A/B Prompt Evaluation

Two judge prompt variants were evaluated on a 12-case failure set.

#### Variant A

```json
{
  "accuracy": 0.5,
  "avg_citation_hit_rate": 1.0,
  "flag_rate": 0.0
}
```

#### Variant B (stricter)

```json
{
  "accuracy": 0.5,
  "avg_citation_hit_rate": 0.75,
  "flag_rate": 0.1667
}
```

### Interpretation

- Variant B applies stricter safety constraints
- Increased guardrail activation reflects more conservative medical reasoning
- Accuracy alone is insufficient.

---

## Challenges and Design Trade-offs

### 1. Chunking Strategy

Choosing an appropriate chunk size was one of the main challenges.

- Very small chunks lost scientific context
- Very large chunks reduced retrieval precision

**Approach**
Documents were chunked at semantically meaningful section boundaries instead of fixed token sizes, balancing retrieval quality and citation accuracy.

---

### 2. Separation of Retrieval and Judgment

Early versions allowed the LLM to compensate for weak retrieval.

**Issue**
This introduced implicit hallucination and blurred responsibility between system components.

**Resolution**
The judge agent is forced to return `Unknown` when evidence is insufficient, making retrieval quality explicit and testable.

---

### 3. Safety vs. Accuracy Trade-off

Stricter prompts reduced unsafe outputs but did not always improve accuracy.

**Observation**
More conservative prompts increased guardrail activation while keeping accuracy similar.

**Decision**
Evaluation metrics were expanded beyond accuracy to include safety-related measures.

---

### 4. OCR Limitations on Real-world Packaging

Supplement labels are often printed on curved bottles with reflective surfaces.

**Issue**
OCR performance degrades when text is distorted by curvature, glare, or complex backgrounds, which can lead to incomplete or noisy extracted text.

This is a known limitation of document-focused OCR models.

**Future Improvements**
- Image preprocessing
- Hybrid vision-language models

---

## Key Takeaways

- Retrieval-grounded reasoning significantly reduces hallucination risk
- Agent separation improves transparency and debuggability
- Evaluation metrics must reflect safety objectives, not only accuracy

---

## Notes

This project is intended for analysis and verification, not medical advice.
All conclusions are strictly limited to cited public evidence.
