# Input -> Retrieval -> Judge

from agents.retriever import retrieve
from agents.judge import judge

def main():
    # CLI input
    claim = input("Enter supplement claim (ad/label text): ").strip()
    if not claim:
        print("Empty input. Exit.")
        return

    # Retrieval
    evidence = retrieve(claim, k=5)

    print("\n[Top evidence]")
    for e in evidence:
        print(f"- score={e['score']:.4f}  {e['chunk_id']}")

    # Evidence Judge + Guardrail
    result, flags = judge(claim, evidence)

    print("\n[Judge output]")
    print(result)

    if flags:
        print("\n[Guardrail flags detected]", flags)

if __name__ == "__main__":
    main()
