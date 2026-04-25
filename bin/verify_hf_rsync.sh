#!/usr/bin/env bash
# Verify the PLM weight mirror from /store → /gpfs. Run after the rsync
# kicked off by this session has had time to finish. Reports to stdout
# (and, when invoked via `systemd-run --user` with a log file, to
# /tmp/plmmsa-hf-rsync-verify.log).
#
# Usage (direct):    ./bin/verify_hf_rsync.sh
# Usage (scheduled): systemd-run --user --on-active=30m bash -lc \
#                      './bin/verify_hf_rsync.sh > /tmp/plmmsa-hf-rsync-verify.log 2>&1'

set -u  # no -e so partial failures still surface every check

SRC=/store/deepfold/huggingface/hub
DST=/gpfs/deepfold/model_cache/hub
REPOS=(
  models--facebook--esm1b_t33_650M_UR50S
  models--Rostlab--prot_t5_xl_uniref50
  models--ElnaggarLab--ankh-large
  models--DeepFoldProtein--Ankh-Large-Contrastive
)

echo "=== plmMSA HF-cache rsync verifier ==="
echo "date:          $(date -Is)"
echo "src:           $SRC"
echo "dst:           $DST"
echo

running=$(pgrep -cf 'rsync.*model_cache/hub' || true)
echo "running rsync processes: $running"
if [[ "$running" -gt 0 ]]; then
  echo "(rsync still in flight — counts below are partial; rerun when pgrep hits 0)"
fi
echo

if [[ ! -d "$SRC" ]]; then
  echo "NOTE: $SRC is not accessible (mount down?). Falling back to dst-only checks."
  SRC_REACHABLE=0
else
  SRC_REACHABLE=1
fi

status=0
for repo in "${REPOS[@]}"; do
  echo "--- $repo ---"
  dst_path="$DST/$repo"
  if [[ ! -d "$dst_path" ]]; then
    echo "  FAIL  dst missing: $dst_path"
    status=1
    continue
  fi

  dst_blobs=$(ls "$dst_path/blobs" 2>/dev/null | wc -l)
  dst_size=$(du -sb "$dst_path" 2>/dev/null | awk '{print $1}')
  dst_h=$(du -sh "$dst_path" 2>/dev/null | awk '{print $1}')
  echo "  dst  blobs=$dst_blobs  size=$dst_h ($dst_size bytes)"

  if [[ "$SRC_REACHABLE" == "1" ]]; then
    src_path="$SRC/$repo"
    if [[ ! -d "$src_path" ]]; then
      echo "  NOTE src missing: $src_path"
      continue
    fi
    src_blobs=$(ls "$src_path/blobs" 2>/dev/null | wc -l)
    src_size=$(du -sb "$src_path" 2>/dev/null | awk '{print $1}')
    src_h=$(du -sh "$src_path" 2>/dev/null | awk '{print $1}')
    echo "  src  blobs=$src_blobs  size=$src_h ($src_size bytes)"

    if [[ "$src_blobs" != "$dst_blobs" ]]; then
      echo "  FAIL  blob count mismatch ($src_blobs vs $dst_blobs)"
      status=1
    fi
    if [[ -n "${src_size:-}" && -n "${dst_size:-}" && "$src_size" != "$dst_size" ]]; then
      diff=$(( src_size - dst_size ))
      abs=${diff#-}
      # tolerate a 1% size drift (sparse files, metadata)
      tol=$(( src_size / 100 ))
      if [[ "$abs" -gt "$tol" ]]; then
        echo "  FAIL  size drift > 1%: |$diff| > $tol bytes"
        status=1
      else
        echo "  ok    size within 1% tolerance"
      fi
    fi

    # Sample-hash: pick the first blob in the src, verify its sha256
    # matches the corresponding dst blob.
    sample=$(ls "$src_path/blobs" 2>/dev/null | head -1 || true)
    if [[ -n "$sample" && -f "$src_path/blobs/$sample" && -f "$dst_path/blobs/$sample" ]]; then
      src_hash=$(sha256sum "$src_path/blobs/$sample" | awk '{print $1}')
      dst_hash=$(sha256sum "$dst_path/blobs/$sample" | awk '{print $1}')
      if [[ "$src_hash" == "$dst_hash" ]]; then
        echo "  ok    sample blob $sample matches ($src_hash)"
      else
        echo "  FAIL  sample blob $sample differs"
        echo "        src=$src_hash"
        echo "        dst=$dst_hash"
        status=1
      fi
    fi
  fi
  echo
done

echo "=== summary ==="
du -sh "$DST" 2>/dev/null
if [[ "$status" == 0 ]]; then
  echo "RESULT: OK"
else
  echo "RESULT: MISMATCH (exit 1)"
fi
exit "$status"
