"""Capture a candidate NEFF with the same inputs used for correctness."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    target = run / "profile"
    target.mkdir(exist_ok=False)
    inputs = torch.load(run / "inputs.pt", map_location="cpu", weights_only=True)
    ifmaps = []
    for name, value in inputs.items():
        path = target / f"{name}.bin"
        path.write_bytes(value.contiguous().view(torch.uint8).numpy().tobytes())
        ifmaps.extend((name, str(path)))
    neff = run / "compile/kernel.neff"
    command = ["/opt/aws/neuron/bin/neuron-explorer", "capture", "-n", str(neff),
               "-s", str(target / "profile.ntff"), "--num-exec=2", "--profile-nth-exec=2",
               "--enable-dge-notifs", *ifmaps]
    environment = dict(os.environ, NEURON_RT_ENABLE_DGE_NOTIFICATIONS="1")
    with (target / "capture.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
    ntffs = sorted(target.glob("*.ntff"))
    if len(ntffs) != 1:
        raise RuntimeError(f"Expected one NTFF, got {ntffs}")
    view = ["/opt/aws/neuron/bin/neuron-explorer", "view", "-n", str(neff), "-s", str(ntffs[0]),
            "--output-format", "summary-json"]
    with (target / "metrics.json").open("w") as output, (target / "view.log").open("w") as log:
        subprocess.run(view, stdout=output, stderr=log, check=True)
    print((target / "metrics.json").read_text())
    (target / "commands.json").write_text(json.dumps({"capture": command, "view": view}, indent=2) + "\n")


if __name__ == "__main__":
    main()
