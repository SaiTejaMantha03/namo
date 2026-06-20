import json
from pathlib import Path

def main():
    project_dir = Path(__file__).resolve().parent.parent
    path_3way = project_dir / "results" / "evaluation_tables" / "snamo_vs_mappo_v3_vs_v4.json"
    path_pure = project_dir / "results" / "evaluation_tables" / "pure_snamo_eval_results.json"

    if not path_3way.exists():
        print(f"File not found: {path_3way}")
        return
    if not path_pure.exists():
        print(f"File not found: {path_pure}")
        return

    with open(path_3way) as f:
        data_3way = json.load(f)

    with open(path_pure) as f:
        data_pure = json.load(f)

    for key, val in data_pure.items():
        if key in data_3way:
            data_3way[key]["pure_snamo"] = {
                "sr": val["sr"],
                "avg_steps": round(val["avg_steps"], 2)
            }
        else:
            print(f"Warning: scenario {key} not found in 3-way table")

    with open(path_3way, "w") as f:
        json.dump(data_3way, f, indent=2)

    print("Successfully merged pure_snamo results into snamo_vs_mappo_v3_vs_v4.json!")

if __name__ == "__main__":
    main()
