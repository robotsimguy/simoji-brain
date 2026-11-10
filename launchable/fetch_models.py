"""Download the GR00T checkpoint and the models it loads, without a Hugging Face token.

nvidia/Cosmos-Reason2-2B (GR00T N1.7's backbone) is gated on Hugging Face. Simoji redistributes an
unmodified copy under the NVIDIA Open Model License (Section 3), and this script places it in the local
Hugging Face cache under the original name, so GR00T loads it offline exactly as if it came from NVIDIA.
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

# Gated upstream id -> public Simoji copy (same files, plus the NVIDIA licence and NOTICE).
MIRRORS = {"nvidia/Cosmos-Reason2-2B": os.environ.get("SIMOJI_COSMOS_REASON_MIRROR", "Simoji/Cosmos-Reason2-2B")}


def put_in_cache(upstream: str, mirror: str, link_dir: str | None = None) -> None:
    """Download mirror and expose it in the HF cache as upstream (refs/main -> same snapshot).

    Also links <link_dir>/<upstream> -> the snapshot: GR00T passes the id to transformers' from_pretrained, which
    treats an existing directory of that (relative) name as a local model and makes no Hub calls. Without it,
    transformers 4.57 asks the Hub about the upstream id even when the files are cached (fails offline)."""
    from huggingface_hub.constants import HF_HUB_CACHE

    src = snapshot_download(mirror)
    repo_dir = os.path.join(HF_HUB_CACHE, "models--" + upstream.replace("/", "--"))
    rev = os.path.basename(src)
    os.makedirs(os.path.join(repo_dir, "refs"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "snapshots"), exist_ok=True)
    link = os.path.join(repo_dir, "snapshots", rev)
    if not os.path.exists(link):
        os.symlink(src, link)
    with open(os.path.join(repo_dir, "refs", "main"), "w") as f:
        f.write(rev)
    if link_dir:
        dst = os.path.join(link_dir, upstream)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.islink(dst):
            os.remove(dst)
        if not os.path.exists(dst):
            os.symlink(src, dst)
    print(f"{upstream}: from {mirror} ({rev[:8]})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--link-dir", default=None, help="the model server's working directory")
    a = ap.parse_args()
    for upstream, mirror in MIRRORS.items():
        put_in_cache(upstream, mirror, a.link_dir)
    print(a.model, snapshot_download(a.model), flush=True)


if __name__ == "__main__":
    main()
