# Agent 2: Evidence Judge + Guardrail (LLM)

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

def _build_prompt(claim: str, evidence):
    block = ""
    for e in evidence:
        block += (
            f"\n---\n"
            f"CITATION: {e['chunk_id']} ({e['doc_id']})\n"
            f"TEXT:\n{e['text']}\n"
        )

    prompt = f"""
You are an evidence judge for supplement claims.

RULES:
- Use ONLY the evidence below.
- If evidence is insufficient, verdict = "Unknown".
- No diagnosis, no dosage, no medical advice.
- Output JSON only.

Claim:
{claim}

Evidence:
{block}

Return JSON:
{{
  "verdict": "Supported" | "Mixed" | "NotSupported" | "Unknown",
  "short_reason": "1-2 evidence-based sentences",
  "citations": ["chunk_id1", "chunk_id2"]
}}
""".strip()

    return prompt

def _guardrail(text: str):
    banned = ["dosage", "mg", "take", "cure", "treat", "diagnose"]
    low = text.lower()
    return [w for w in banned if w in low]

def judge(claim: str, evidence):
    #Run LLM judgment and guardrails
    prompt = _build_prompt(claim, evidence)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    output = r.json()["response"]

    flags = _guardrail(output)
    return output, flags
