-- one table for each pipeline run

CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,

  -- user input
  input_mode TEXT NOT NULL,
  claim TEXT NOT NULL,

  -- retrieval info
  topk INTEGER NOT NULL,
  evidence_json TEXT NOT NULL,

  -- judge output (JSON-only)
  judge_json TEXT NOT NULL,

  -- guardrail flags
  flags_json TEXT NOT NULL
);
