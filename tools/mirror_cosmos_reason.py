"""Mirror nvidia/Cosmos-Reason2-2B (gated) to a public Simoji repo, unmodified, with the NVIDIA licence + NOTICE.

Redistribution under the NVIDIA Open Model License, Section 3 (copy of the Agreement + NOTICE file).
Needs a Hugging Face token that has accepted the upstream licence and can write to the target org.

    python tools/mirror_cosmos_reason.py --to Simoji/Cosmos-Reason2-2B
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi, snapshot_download

SRC = "nvidia/Cosmos-Reason2-2B"
LICENSE_URL = "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/"

NOTICE = """Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

This repository is an unmodified copy of nvidia/Cosmos-Reason2-2B
(https://huggingface.co/nvidia/Cosmos-Reason2-2B), redistributed by Simoji under Section 3 of the
NVIDIA Open Model License Agreement (copy in LICENSE_NVIDIA_OPEN_MODEL.md) so that the Simoji robot
brain (https://github.com/robotsimguy/simoji-brain) can load NVIDIA Isaac GR00T N1.7 without a
Hugging Face token. Built on NVIDIA Cosmos. Simoji is not affiliated with or endorsed by NVIDIA.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", default="Simoji/Cosmos-Reason2-2B")
    ap.add_argument("--work", default=os.path.expanduser("~/simoji-mirror/Cosmos-Reason2-2B"))
    a = ap.parse_args()
    api = HfApi()
    rev = api.model_info(SRC).sha
    print(f"downloading {SRC} @ {rev[:8]} -> {a.work}", flush=True)
    snapshot_download(SRC, revision=rev, local_dir=a.work)

    readme = os.path.join(a.work, "README.md")
    with open(readme) as f:
        card = f.read()
    # Keep NVIDIA's model card (and its licence metadata) but drop the click-through gate: the licence is
    # passed on as a file instead, as Section 3.1 requires.
    if card.startswith("---"):
        end = card.index("\n---", 3)
        meta = [l for l in card[3:end].splitlines()]
        out, skip = [], False
        for l in meta:
            if l.startswith("extra_gated"):
                skip = True
                continue
            if skip and (l.startswith(" ") or l.startswith("\t") or not l.strip()):
                continue
            skip = False
            out.append(l)
        card = "---" + "\n".join(out) + card[end:]
    banner = (f"> **Simoji mirror.** Unmodified copy of [{SRC}](https://huggingface.co/{SRC}) at revision "
              f"`{rev}`, redistributed under the NVIDIA Open Model License (see `NOTICE` and "
              "`LICENSE_NVIDIA_OPEN_MODEL.md`). Built on NVIDIA Cosmos.\n\n")
    head_end = card.index("\n---", 3) + 4 if card.startswith("---") else 0
    card = card[:head_end] + "\n" + banner + card[head_end:].lstrip("\n")
    with open(readme, "w") as f:
        f.write(card)
    with open(os.path.join(a.work, "NOTICE"), "w") as f:
        f.write(NOTICE)
    # The full Agreement text exactly as NVIDIA ships it in the model card's click-through gate.
    agreement = (api.model_info(SRC, revision=rev, expand=["cardData"]).card_data or {}).get("extra_gated_prompt")
    if not agreement:
        raise SystemExit("could not read the licence text from the model card; not uploading")
    with open(os.path.join(a.work, "LICENSE_NVIDIA_OPEN_MODEL.md"), "w") as f:
        f.write(f"<!-- Source: {LICENSE_URL} (as shipped in {SRC}'s model card) -->\n\n{agreement.strip()}\n")

    api.create_repo(a.to, repo_type="model", exist_ok=True, private=False)
    print(f"uploading to {a.to}", flush=True)
    api.upload_folder(repo_id=a.to, folder_path=a.work, ignore_patterns=[".cache/*"],
                      commit_message=f"Mirror of {SRC}@{rev[:8]} (NVIDIA Open Model License) + NOTICE")
    print(f"done: https://huggingface.co/{a.to}", flush=True)


if __name__ == "__main__":
    main()
