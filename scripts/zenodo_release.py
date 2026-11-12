#!/usr/bin/env python
"""Build the Zenodo study bundle of a release and upload it as a new version.

    python scripts/zenodo_release.py bundle  --tag v2.0.0
    python scripts/zenodo_release.py draft   --tag v2.0.0 --previous 23484768
    python scripts/zenodo_release.py publish --deposition <id>

``bundle`` writes ``dist/xai-gov-bundle-<tag>.zip`` from ``outputs/v2`` and the
tracked files of the tag: every run, the campaign reports and cells, the
sweeps, the detector null line, Theorem A1, the figures, the calibration
reports, ``configs/`` and ``specs/v2/``, plus ``PROVENANCE.txt`` (repository,
tag, commit) and ``SHA256SUMS`` (one line per file). The M5 series are never
included: their licence does not permit redistribution.

``draft`` opens a new version of the previous record, keeps its creators and
licence, replaces the title, description, version, keywords, date and links
with ``specs/v2/zenodo_<tag>.json``, removes the inherited files and uploads the
bundle. It does **not** publish: the draft can be checked on zenodo.org first.

``publish`` mints the DOI. It cannot be undone, so it is a separate command.

The token is read from ``$ZENODO_TOKEN`` or from ``~/.config/zenodo/token``. It
needs the scopes ``deposit:write`` and ``deposit:actions``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
API = "https://zenodo.org/api"
REPOSITORY = "https://github.com/rmachuchoc/xai-gov"
TOKEN_FILE = Path.home() / ".config" / "zenodo" / "token"

#: (source relative to the repository, name inside the bundle)
OUTPUT_TREES = (
    ("outputs/v2/runs", "runs"),
    ("outputs/v2/campaigns", "campaigns"),
    ("outputs/v2/sweeps", "sweeps"),
    ("outputs/v2/analysis", "analysis"),
    ("outputs/v2/figures", "figures"),
    ("outputs/calibration", "calibration"),
)
#: Tracked trees and files, read from the tag rather than the working copy.
TRACKED = ("configs", "specs/v2", "README.md", "LICENSE")
CHUNK = 1 << 20


# --------------------------------------------------------------------- bundle
def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()


def tracked_files(tag: str) -> list[tuple[str, bytes]]:
    """(path, content) of the tracked files at the tag, not the working copy."""
    names = git("ls-tree", "-r", "--name-only", tag, "--", *TRACKED).splitlines()
    names.append(f"CHANGELOG-{tag}.md")
    out = []
    for name in names:
        blob = subprocess.run(["git", "show", f"{tag}:{name}"], cwd=ROOT, check=True,
                              capture_output=True).stdout
        out.append((name, blob))
    return out


def output_files() -> list[tuple[Path, str]]:
    files = []
    for source, target in OUTPUT_TREES:
        base = ROOT / source
        if not base.is_dir():
            raise SystemExit(f"missing {source}")
        for path in sorted(base.rglob("*")):
            if path.is_file():
                files.append((path, f"{target}/{path.relative_to(base).as_posix()}"))
    return files


def build_bundle(tag: str, out: Path) -> Path:
    commit = git("rev-list", "-n", "1", tag)
    prefix = f"xai-gov-bundle-{tag}"
    sums: list[str] = []
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as bundle:
        outputs = output_files()
        for i, (path, name) in enumerate(outputs, 1):
            digest = hashlib.sha256()
            with path.open("rb") as src, bundle.open(f"{prefix}/{name}", "w") as dst:
                while chunk := src.read(CHUNK):
                    digest.update(chunk)
                    dst.write(chunk)
            sums.append(f"{digest.hexdigest()}  {name}")
            if i % 5000 == 0:
                print(f"  {i}/{len(outputs)} output files", flush=True)
        for name, blob in tracked_files(tag):
            bundle.writestr(f"{prefix}/{name}", blob)
            sums.append(f"{hashlib.sha256(blob).hexdigest()}  {name}")
        provenance = (
            f"repository: {REPOSITORY}\n"
            f"tag: {tag}\n"
            f"commit: {commit}\n"
            f"built: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
            f"files: {len(sums)} (checksums in SHA256SUMS)\n"
            "not included: M5 demand series (licence); regenerate with `make m5`\n"
        )
        bundle.writestr(f"{prefix}/PROVENANCE.txt", provenance)
        ordered = sorted(sums, key=lambda line: line[66:])  # by file name
        bundle.writestr(f"{prefix}/SHA256SUMS", "\n".join(ordered) + "\n")
    partial.replace(out)
    print(f"{out}: {len(sums)} files, {out.stat().st_size / 1e6:.1f} MB, commit {commit[:12]}")
    return out


# --------------------------------------------------------------------- zenodo
def token() -> str:
    value = os.environ.get("ZENODO_TOKEN") or (
        TOKEN_FILE.read_text(encoding="utf-8").strip() if TOKEN_FILE.is_file() else "")
    if not value:
        raise SystemExit(f"no token: set ZENODO_TOKEN or write it to {TOKEN_FILE}")
    return value


def call(method: str, url: str, body: Any = None, *, raw: Any = None,
         size: int | None = None) -> Any:
    headers = {"Authorization": f"Bearer {token()}", "User-Agent": "xai-gov-release"}
    data: Any = None
    if raw is not None:
        data = raw
        headers["Content-Type"] = "application/octet-stream"
        headers["Content-Length"] = str(size)
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{method} {url}: {error.code} {error.read().decode()[:800]}") from None
    return json.loads(payload) if payload else None


def draft(tag: str, previous: int, bundle: Path, metadata_file: Path) -> None:
    if not bundle.is_file():
        raise SystemExit(f"{bundle} not found: run the bundle command first")
    overrides = json.loads(metadata_file.read_text(encoding="utf-8"))
    opened = call("POST", f"{API}/deposit/depositions/{previous}/actions/newversion")
    deposition = call("GET", opened["links"]["latest_draft"])
    dep_id = deposition["id"]
    for f in deposition.get("files", []):
        call("DELETE", f"{API}/deposit/depositions/{dep_id}/files/{f['id']}")
    metadata = deposition["metadata"]
    metadata.update(overrides)
    metadata["version"] = tag
    metadata["publication_date"] = datetime.now(UTC).date().isoformat()
    metadata.pop("doi", None)
    updated = call("PUT", f"{API}/deposit/depositions/{dep_id}", {"metadata": metadata})
    reserved = updated["metadata"].get("prereserve_doi", {}).get("doi")
    size = bundle.stat().st_size
    print(f"uploading {bundle.name} ({size / 1e6:.1f} MB) to draft {dep_id} ...", flush=True)
    with bundle.open("rb") as handle:
        uploaded = call("PUT", f"{deposition['links']['bucket']}/{bundle.name}", raw=handle,
                        size=size)
    digest = hashlib.md5(usedforsecurity=False)
    with bundle.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    remote = str(uploaded.get("checksum", "")).removeprefix("md5:")
    if remote != digest.hexdigest():
        raise SystemExit(f"checksum mismatch: local {digest.hexdigest()}, Zenodo {remote}")
    print(f"draft {dep_id} ready, checksum verified")
    print(f"  review: https://zenodo.org/deposit/{dep_id}")
    print(f"  reserved DOI: {reserved}")
    print(f"  publish with: python scripts/zenodo_release.py publish --deposition {dep_id}")


def publish(dep_id: int) -> None:
    record = call("POST", f"{API}/deposit/depositions/{dep_id}/actions/publish")
    print(f"published: https://doi.org/{record['doi']}  ({record['links']['html']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Zenodo study bundle")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bundle", help="build the zip")
    b.add_argument("--tag", required=True)
    d = sub.add_parser("draft", help="new version of a record, unpublished")
    d.add_argument("--tag", required=True)
    d.add_argument("--previous", type=int, required=True, help="id of the latest record")
    p = sub.add_parser("publish", help="mint the DOI (irreversible)")
    p.add_argument("--deposition", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "bundle":
        build_bundle(args.tag, ROOT / "dist" / f"xai-gov-bundle-{args.tag}.zip")
    elif args.command == "draft":
        draft(args.tag, args.previous, ROOT / "dist" / f"xai-gov-bundle-{args.tag}.zip",
              ROOT / "specs" / "v2" / f"zenodo_{args.tag}.json")
    else:
        publish(args.deposition)
    return 0


if __name__ == "__main__":
    sys.exit(main())
