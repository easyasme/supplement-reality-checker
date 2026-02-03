# Input -> (optional OCR) -> Retrieval -> Judge => telemetry logging
import json

from telemetry.db import log_run
from agents.retriever import retrieve
from agents.judge import judge
from agents.vision_extractor import extract_label_text

def main():
    print("Choose input type:")
    print("1) text (paste claim/label text)")
    print("2) image (OCR label image)")
    mode = input("Enter 1 or 2: ").strip()

    claim = ""
    input_mode = "text"

    # input collection (text or image)
    if mode == "2":
        input_mode = "image"
        image_path = input("Image path: ").strip()

        # image -> label text (HF OCR)
        claim = extract_label_text(image_path)

        print("\n[OCR extracted text]")
        print(claim)

    else:
        # normal text input
        claim = input("Enter supplement label text / ad text: ").strip()

    if not claim:
        print("Empty input. Exit.")
        return

    # Retrieval (Agent 1)
    k = 5 # top-k
    evidence = retrieve(claim, k=k)

    print("\n[Top evidence]")
    for e in evidence:
        print(f"- score={e['score']:.4f}  {e['chunk_id']}  ({e['doc_id']})")

    # Evidence Judge + Guardrail (Agent 2)
    result_obj, flags = judge(claim, evidence)
    print("\n[Judge output JSON]")
    print(json.dumps(result_obj, indent=2, ensure_ascii=False))

    if flags:
        print("\n[Guardrail flags detected]", flags)
    
    log_run(
        input_mode=input_mode,
        claim=claim,
        topk=k,
        evidence=evidence,
        judge_obj=result_obj,
        flags=flags,
    )

    print("\n[Telemetry] Saved to telemetry/telemetry.db")

if __name__ == "__main__":
    main()
