#!/bin/bash
# push_release.sh — Stage and upload model weights to a GitHub Release.
#
# The weight files live in ../release/ (git-ignored, too large for plain git).
# This script:
#   1. Verifies every file in release/ matches weights_manifest.json (SHA256)
#   2. Creates (or reuses) a GitHub release with the given tag
#   3. Uploads the files flagged github_release_asset=true in the manifest
#   4. Uploads .sha256 sidecar files for independent verification
#
# Prerequisites:
#   - gh CLI installed and authenticated (gh auth login)
#   - Files present in release/ (run scripts/fetch_weights_from_hf.sh first)
#
# Usage:
#   ./scripts/push_release.sh <tag> [release-title]
#
# Example:
#   ./scripts/push_release.sh v0.1.0-models "Model weights v0.1.0"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RELEASE_DIR="$REPO_ROOT/release"
MANIFEST="$RELEASE_DIR/weights_manifest.json"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <tag> [release-title]"
    echo "Example: $0 v0.1.0-models \"Model weights v0.1.0\""
    exit 1
fi

TAG="$1"
TITLE="${2:-Model weights $TAG}"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Manifest not found at $MANIFEST"
    echo "Run scripts/fetch_weights_from_hf.sh first to populate release/."
    exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: gh CLI not found. Install from https://cli.github.com/ and run 'gh auth login'."
    exit 1
fi

echo "=== Step 1: Verify release/ files against manifest ==="
cd "$RELEASE_DIR"

# Use python3 to parse the manifest and verify each file
python3 - "$MANIFEST" <<'PYEOF'
import json, hashlib, os, sys

manifest_path = sys.argv[1]
with open(manifest_path) as f:
    manifest = json.load(f)

errors = 0
for fname, info in manifest["files"].items():
    expected_sha = info["sha256"]
    expected_size = info["size"]
    if not os.path.exists(fname):
        print(f"  ❌ MISSING: {fname}")
        errors += 1
        continue
    actual_size = os.path.getsize(fname)
    if actual_size != expected_size:
        print(f"  ❌ SIZE MISMATCH: {fname} (expected {expected_size}, got {actual_size})")
        errors += 1
        continue
    h = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    actual_sha = h.hexdigest()
    if actual_sha != expected_sha:
        print(f"  ❌ SHA256 MISMATCH: {fname}")
        print(f"     expected: {expected_sha}")
        print(f"     actual:   {actual_sha}")
        errors += 1
    else:
        size_mb = actual_size / (1024 * 1024)
        print(f"  ✅ {fname} ({size_mb:.1f} MB)")

if errors:
    print(f"\n{errors} file(s) failed verification. Aborting.")
    sys.exit(1)
else:
    print("\nAll files verified. Ready to upload.")
PYEOF

echo ""
echo "=== Step 2: Create GitHub release '$TAG' (if it doesn't exist) ==="
if gh release view "$TAG" >/dev/null 2>&1; then
    echo "Release '$TAG' already exists. Files will be added/updated."
else
    gh release create "$TAG" \
        --title "$TITLE" \
        --notes "Model weights for moss-tts-nano-rust-candle.

Mirrored from HuggingFace ramishi/moss-tts-nano-candle @ $(python3 -c "import json;print(json.load(open('$MANIFEST'))['hf_revision'])").

See release/weights_manifest.json in the repo for SHA256 verification." \
        --generate-notes
    echo "Created release '$TAG'."
fi

echo ""
echo "=== Step 3: Upload release assets (github_release_asset=true) ==="
# Generate .sha256 sidecar files and upload assets flagged in the manifest
ASSETS=()
SIDECARS=()
while IFS= read -r fname; do
    if [ ! -f "$fname" ]; then
        echo "  ⚠️  Skipping $fname (not present)"
        continue
    fi
    # Generate .sha256 sidecar
    sha256sum "$fname" | awk '{print $1}' > "$fname.sha256"
    ASSETS+=("$fname" "$fname.sha256")
    echo "  + $fname (+ .sha256 sidecar)"
done < <(python3 -c "
import json
with open('$MANIFEST') as f:
    m = json.load(f)
for fname, info in m['files'].items():
    if info.get('github_release_asset'):
        print(fname)
")

if [ ${#ASSETS[@]} -eq 0 ]; then
    echo "No assets to upload."
    exit 0
fi

gh release upload "$TAG" "${ASSETS[@]}" --clobber

echo ""
echo "=== Step 4: Verify uploaded assets ==="
gh release view "$TAG" --json assets --jq '.assets[] | "\(.name)\t\(.size) bytes"' | while read -r line; do
    echo "  📦 $line"
done

echo ""
echo "✅ Done. Release '$TAG' updated."
echo "   Download URL pattern:"
echo "   https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/download/$TAG/<filename>"
