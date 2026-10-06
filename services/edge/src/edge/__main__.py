"""``python -m edge`` — the edge container's default command (see Dockerfile).

The edge has no server of its own; it is a set of command-line stages. This
lists them so the container's default command does something useful instead of
failing with "No module named edge.__main__".
"""

ENTRY_POINTS = (
    ("edge.train_client", "train a client LoRA (D3: --cluster-adapter, D7: --dp)"),
    ("edge.merge", "compose base + alpha*cluster + beta*client into one adapter (D6)"),
    ("edge.round", "alpha sweep + held-out eval for a whole round"),
    ("edge.registry_client", "pull an adapter out of the registry (seam C1)"),
    ("edge.completion_eval", "in-project next-line completion metric (D5)"),
    ("edge.ttft", "TTFT / adapter-swap benchmark (D11)"),
    ("edge.aggregate", "D2 aggregation, computed by P2's cluster package"),
)


def main() -> None:
    print("CLASP edge (P1). Stages — run with `python -m <module> --help`:")
    for module, what in ENTRY_POINTS:
        print(f"  {module:22s} {what}")


if __name__ == "__main__":
    main()
