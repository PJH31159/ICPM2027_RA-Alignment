"""
validation_utils — 검사 기록, 단계 상태, 로그, 환경 정보, checksum (REF-001)

외부 라이브러리 없이 표준 라이브러리만 쓴다 (어느 환경에서도 import 가능).
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
# 검사 기록
# ------------------------------------------------------------

class CheckRecorder:
    """check()로 통과/실패를 기록하고, 마지막에 상태를 JSON으로 남긴다."""

    def __init__(self):
        self.checks = []

    def check(self, name, ok, detail=""):
        self.checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        return bool(ok)

    def info(self, name, value):
        """통과/실패가 아닌 관찰값 (예: Medium의 generated-only 수)."""
        self.checks.append({"check": name, "passed": None, "detail": str(value)})
        print(f"  [INFO] {name} {value}")

    @property
    def failed(self):
        return [c for c in self.checks if c["passed"] is False]

    @property
    def ok(self):
        return not self.failed


# ------------------------------------------------------------
# 단계 상태 (다음 단계는 이전 단계가 PASSED여야 시작한다)
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
# 로그 (화면 + 파일)
# ------------------------------------------------------------

class Tee:
    """with Tee(path): 블록 안의 print를 화면과 파일에 함께 쓴다."""

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
# 환경 / 파일 정보
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
# 과거 실험 script 보존 확인 (REF-002)
# ------------------------------------------------------------

# REF-002 이동 때 허용된 변경: 프로젝트 root 계산, sys.path 추가, 같은 폴더 경로, 실행 예시 경로
_REF002_REMOVED_OK = (
    "    .parent",
    "PROJECT_ROOT = Path(__file__).resolve().parent.parent",
    'EXP061_SCRIPT = PROJECT_ROOT / "src" / "61_method_b_alignment_repair.py"',
    '    spec = importlib.util.spec_from_file_location("exp062", PROJECT_ROOT / "src" / "62_compare_method_a_b.py")',
    "from sim import run_rpbear",
)
_REF002_ADDED_OK = ("", "import sys", "from pathlib import Path", "from sim import run_rpbear  # noqa: E402")


def historical_script_drift(new_rel, old_rel):
    """과거 실험 script가 HEAD 대비 REF-002 경로 수정 외에 바뀌었는지 확인한다.

    HEAD에 새 경로가 있으면(=REF-002 커밋 후) 그 파일과 완전히 같아야 한다.
    없으면(=커밋 전) HEAD의 옛 경로 파일과 비교하고, 허용된 경로 수정 줄만 다를 수 있다.
    return: 예상하지 못한 변경 줄 목록 (빈 목록이면 보존됨)
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
