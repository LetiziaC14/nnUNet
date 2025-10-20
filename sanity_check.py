import os, json, sys, traceback

print("Python:", sys.executable)
root = os.getenv("nnUNet_raw")
print("nnUNet_raw:", root)

try:
    if not root:
        raise RuntimeError("nnUNet_raw is not set in this VS Code terminal/session.")

    p = os.path.join(root, "Dataset223_MassesTumor", "dataset.json")
    print("Reading:", p)
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {p}")

    with open(p, "r") as f:
        d = json.load(f)

    # Verbose echo of labels field
    print("labels field:", d.get("labels"))

    assert isinstance(d["labels"]["kidney"], list),  "'kidney' must be a list"
    assert isinstance(d["labels"]["masses"], list),  "'masses' must be a list"
    assert isinstance(d["labels"]["tumor"], list),   "'tumor' must be a list"
    assert all(isinstance(x, int) for x in d["labels"]["tumor"]), "'tumor' entries must be ints"
    print("OK: regions schema looks correct.")

except Exception as e:
    print("ERROR:", e)
    traceback.print_exc()

