"""
Parse scan_openmax_params.py output files, extract best AUROC per dataset,
then update web-frontend/backend/app.py with real values.
"""
import argparse
import json
import re
import os

APP_PY = os.path.join(os.path.dirname(__file__), "web-frontend", "backend", "app.py")

# NSL-KDD known result (already computed)
NSLKDD_AUROC = 0.964785651641781


def parse_best(path):
    """Return (auroc, aupr_out, best_balanced_acc, tail, alpha, dist) from scan output."""
    if not os.path.isfile(path):
        raise FileNotFoundError("Scan output not found: {}".format(path))
    best_auroc = None
    best_line = None
    with open(path) as fh:
        in_top = False
        for line in fh:
            if "Top " in line and "results by AUROC" in line:
                in_top = True
                continue
            if in_top and re.match(r"^\s*1\.", line):
                best_line = line.strip()
                break
    if best_line is None:
        # fall back: scan every printed line for AUROC=
        aurocs = []
        with open(path) as fh:
            for line in fh:
                m = re.search(r"AUROC=([0-9.]+)", line)
                if m:
                    aurocs.append(float(m.group(1)))
        if not aurocs:
            raise ValueError("No AUROC found in {}".format(path))
        best_auroc = max(aurocs)
        return best_auroc, None, None, None, None, None

    nums = {k: v for k, v in re.findall(r"(\w+)=([0-9.]+)", best_line)}
    dist_m = re.search(r"dist=(\w+)", best_line)
    dist = dist_m.group(1) if dist_m else None
    return (
        float(nums.get("AUROC", 0)),
        float(nums.get("AUPR_OUT", 0)),
        float(nums.get("BAL_ACC", 0)),
        int(nums.get("tail", 0)),
        int(nums.get("alpha", 0)),
        dist,
    )


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cicids",     required=True)
    p.add_argument("--cicids2018", required=True)
    p.add_argument("--unsw",       required=True)
    return p.parse_args()


def main():
    args = get_args()

    results = {}
    for key, path, label in [
        ("cicids",     args.cicids,     "CICIDS-2017"),
        ("cicids2018", args.cicids2018, "CICIDS-2018"),
        ("unsw_nb15",  args.unsw,       "UNSW-NB15"),
    ]:
        auroc, aupr, bal_acc, tail, alpha, dist = parse_best(path)
        results[key] = auroc
        print("{}: AUROC={:.6f}  (tail={} alpha={} dist={})".format(
            label, auroc, tail, alpha, dist))

    results["nslkdd"] = NSLKDD_AUROC
    print("NSL-KDD:    AUROC={:.6f}  (pre-computed)".format(NSLKDD_AUROC))

    # Save JSON for reference
    out_json = os.path.join(os.path.dirname(__file__), "best_auroc_results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved to", out_json)

    # Patch app.py
    _patch_app(results)
    print("Updated", APP_PY)


def _patch_app(results):
    with open(APP_PY) as fh:
        src = fh.read()

    # Replace each auroc value in DATASET_RESULTS
    replacements = {
        "'cicids'":     results["cicids"],
        "'cicids2018'": results["cicids2018"],
        "'nslkdd'":     results["nslkdd"],
        "'unsw_nb15'":  results["unsw_nb15"],
    }

    for key, auroc_val in replacements.items():
        # Match the block for this dataset key and replace the 'auroc' field
        pattern = r"({key}:\s*\{{[^}}]*?'auroc':\s*)[0-9.]+".format(key=re.escape(key))
        replacement = r"\g<1>{:.4f}".format(auroc_val)
        src, n = re.subn(pattern, replacement, src, flags=re.DOTALL)
        if n == 0:
            print("WARNING: could not patch auroc for {}".format(key))

    with open(APP_PY, "w") as fh:
        fh.write(src)


if __name__ == "__main__":
    main()
