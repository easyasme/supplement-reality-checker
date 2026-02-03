# Agent 2: Evidence Judge + Guardrail (Ollama)
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

def build_prompt(claim: str, evidence, variant: str = "A") -> str:
    """
    variant = "A" or "B"
    - A: normal strict "use only evidence"
    - B: even stricter: if evidence doesn't mention claim topic -> Unknown
    """
    block = ""
    for e in evidence:
        block += (
            "\n---\n"
            f"CITATION: {e['chunk_id']} ({e['doc_id']})\n"
            f"TEXT:\n{e['text']}\n"
        )

    # prompt A
    prompt_a = f"""
You are an evidence judge for supplement claims.

RULES:
- Use ONLY the evidence below.
- If evidence is insufficient, verdict = "Unknown".
- No diagnosis, no dosage, no medical advice.
- Output JSON only (no markdown, no extra text).

Claim:
{claim}

Evidence:
{block}

Return JSON:
{{
  "verdict": "Supported" | "Mixed" | "NotSupported" | "Unknown",
  "short_reason": "1-2 evidence-based sentences",
  "citations": ["chunk_id1", "chunk_id2"],
  "confidence": 0.0
}}
""".strip()

    # prompt B (strict mode)
    prompt_b = f"""
You are an evidence judge for supplement claims.

STRICT RULES:
- Use ONLY the evidence below.
- If the evidence does NOT explicitly discuss the claim topic/effect, verdict MUST be "Unknown".
- Never guess. Never generalize beyond the text.
- No diagnosis, no dosage, no medical advice.
- Output JSON only (no markdown, no extra text).

Claim:
{claim}

Evidence:
{block}

Return JSON:
{{
  "verdict": "Supported" | "Mixed" | "NotSupported" | "Unknown",
  "short_reason": "1-2 evidence-based sentences",
  "citations": ["chunk_id1", "chunk_id2"],
  "confidence": 0.0
}}
""".strip()

    return prompt_b if variant.upper() == "B" else prompt_a

def guardrail_flags(text: str):
    # simple guardrails
    banned = ["dosage", "mg", "take", "cure", "treat", "diagnose", "prescribe"]
    low = text.lower()
    return [w for w in banned if w in low]

def _safe_json_parse(s: str):
    s = s.strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to find first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = s[start:end+1]
        return json.loads(chunk)

    raise ValueError("Could not parse JSON from model output.")

def judge(claim: str, evidence, variant: str = "A"):
    """
    - variant: "A" or "B"
    - returns (judge_obj, flags)
    """
    prompt = build_prompt(claim, evidence, variant=variant)

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

    raw = r.json()["response"]

    judge_obj = _safe_json_parse(raw)

    flags = guardrail_flags(json.dumps(judge_obj, ensure_ascii=False))
    return judge_obj, flags
