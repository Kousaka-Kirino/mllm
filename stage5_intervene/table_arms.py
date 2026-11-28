"""把各臂的四格与总准确率打出来（RQ2 收尾用，一次性工具）。"""

import json
import sys

RUNS = sys.argv[1:] or ["sft_fulltrain_dev", "arm_plain",
                        "arm_curriculum", "arm_counterfactual", "arm_grpo"]
CELLS = ("easy_consistent", "easy_conflict", "hard_consistent", "hard_conflict")


def rows_of(run):
    try:
        return json.load(open(f"outputs/eval/{run}/records.json", encoding="utf-8"))
    except FileNotFoundError:
        return None


def main() -> int:
    print(f"{'run':22s} {'cond':17s} " + " ".join(f"{c[:12]:>8s}" for c in CELLS)
          + f" {'overall':>8s}")
    for run in RUNS:
        rows = rows_of(run)
        if rows is None:
            print(f"{run:22s} (missing)")
            continue
        for cond in ("correct_image", "mismatched_image"):
            by = {r["suite"]: r for r in rows if r["condition"] == cond}
            if not by:
                continue
            cells = " ".join(f"{by[c]['accuracy']:7.3f}" if c in by else "      -"
                             for c in CELLS)
            ov = by.get("overall")
            print(f"{run:22s} {cond:17s} {cells} "
                  f"{ov['accuracy']:8.3f}" if ov else f"{run:22s} {cond:17s} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
