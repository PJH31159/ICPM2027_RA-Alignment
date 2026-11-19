"""Check records, stage status, logging, environment info and checksums.

Standard library only, so it can be imported in every environment.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------
# Check records
# ------------------------------------------------------------

class CheckRecorder:
    """Record pass/fail with check() and write the final status as JSON."""

    def __init__(self):
        self.checks = []

    def check(self, name, ok, detail=""):
        self.checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        return bool(ok)

    def info(self, name, value):
        """An observation that is not pass/fail (e.g. a count)."""
        self.checks.append({"check": name, "passed": None, "detail": str(value)})
        print(f"  [INFO] {name} {value}")

    @property
    def failed(self):
        return [c for c in self.checks if c["passed"] is False]

    @property
    def ok(self):
        return not self.failed


# ------------------------------------------------------------
# Stage status (a stage starts only if the previous stage PASSED)
# ------------------------------------------------------------

STATUS_FILE = "stage_status.json"


def write_stage_status(stage_dir, stage, status, recorder, extra=None):
    payload = {
        "stage": stage,
        "status": status,  # PASSED / FAILED / BLOCKED
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "checks": recorder.checks,
        "environment": environment_info(),
    }
    if extra:
        payload.update(extra)
    write_json(Path(stage_dir) / STATUS_FILE, payload)
    print(f"\n[{status}] stage={stage}")
    for c in recorder.failed:
        print(f"   - FAIL: {c['check']}: {c['detail']}")


def read_stage_status(stage_dir):
    path = Path(stage_dir) / STATUS_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------
# Logging (screen + file)
# ------------------------------------------------------------

class Tee:
    """with Tee(path): write print output of the block to the screen and to a file."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self
        return self

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __exit__(self, *exc):
        sys.stdout = self.stdout
        self.file.close()
        return False


# ------------------------------------------------------------
# Environment / file information
# ------------------------------------------------------------

def environment_info():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"(git error: {e})"

    versions = {}
    for mod in ["pandas", "numpy", "networkx", "pm4py"]:
        m = sys.modules.get(mod)
        versions[mod] = getattr(m, "__version__", None) if m else None
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages_loaded": versions,
        "git_commit": git("rev-parse", "--short", "HEAD"),
    }


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_record(role, path):
    path = Path(path)
    return {"role": role, "resolved_path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_ids(path):
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def project_path(relative):
    return PROJECT_ROOT / relative


# ------------------------------------------------------------
# Preservation check of historical experiment scripts
# ------------------------------------------------------------

# changes allowed when the scripts were moved: project-root computation, sys.path insert, same-folder paths, example paths
_REF002_REMOVED_OK = (
    "    .parent",
    "PROJECT_ROOT = Path(__file__).resolve().parent.parent",
    'EXP061_SCRIPT = PROJECT_ROOT / "src" / "61_method_b_alignment_repair.py"',
    '    spec = importlib.util.spec_from_file_location("exp062", PROJECT_ROOT / "src" / "62_compare_method_a_b.py")',
    "from sim import run_rpbear",
)
_REF002_ADDED_OK = ("", "import sys", "from pathlib import Path", "from sim import run_rpbear  # noqa: E402")


def historical_script_drift(new_rel, old_rel):
    """Check whether a historical experiment script changed relative to HEAD beyond the allowed path edits.

    If the new path exists in HEAD, the file must be identical to it; otherwise it is compared with the
    old-path file in HEAD and only the allowed path lines may differ.
    Returns the list of unexpected changed lines (empty = preserved).
    """
    import difflib

    def head(rel):
        r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=PROJECT_ROOT, capture_output=True)
        return r.stdout.decode("utf-8").replace("\r\n", "\n").splitlines() if r.returncode == 0 else None

    new = (PROJECT_ROOT / new_rel).read_text(encoding="utf-8").replace("\r\n", "\n").splitlines()
    base = head(new_rel)
    strict = base is not None
    if base is None:
        base = head(old_rel)
    if base is None:
        return [f"no HEAD version for {new_rel} or {old_rel}"]
    unexpected = []
    for line in difflib.unified_diff(base, new, lineterm="", n=0):
        if line.startswith(("+++", "---", "@@")):
            continue
        body = line[1:]
        if strict:
            unexpected.append(line)
        elif line.startswith("-") and not (body in _REF002_REMOVED_OK or ("src" + "\\" * 2 + "6") in body):
            unexpected.append(line)
        elif line.startswith("+") and not ("REF-002" in body or body in _REF002_ADDED_OK
                                           or "experiments" + "\\" * 2 + "p6_small_repaired_log_comparison" in body):
            unexpected.append(line)
    return unexpected
