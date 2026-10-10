#!/usr/bin/env python3
"""Orquestrador de testes: sincroniza o ambiente (uv), roda a suíte pytest e
imprime um resumo final claro (passou/falhou, contagens, duração)."""

import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ex.: "23 passed, 1 warning in 1.05s" ou "1 failed, 22 passed in 2.47s"
PASSED_RE = re.compile(r"(\d+) passed")
FAILED_RE = re.compile(r"(\d+) failed")
ERROR_RE = re.compile(r"(\d+) error")
DURATION_RE = re.compile(r"in ([\d.]+)s")


def run(cmd: list[str], step: str) -> int:
    print(f"\n{'=' * 60}\n>> {step}\n{'=' * 60}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def run_pytest_and_capture(cmd: list[str]) -> tuple[int, str]:
    print(f"\n{'=' * 60}\n>> Executando testes\n{'=' * 60}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode, result.stdout


def parse_summary(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            passed = PASSED_RE.search(line)
            failed = FAILED_RE.search(line)
            errors = ERROR_RE.search(line)
            duration = DURATION_RE.search(line)
            return {
                "passed": int(passed.group(1)) if passed else 0,
                "failed": int(failed.group(1)) if failed else 0,
                "errors": int(errors.group(1)) if errors else 0,
                "duration": float(duration.group(1)) if duration else 0.0,
            }
    return {"passed": 0, "failed": 0, "errors": 0, "duration": 0.0}


def main() -> int:
    start = time.time()

    sync_code = run(["uv", "sync", "--group", "test"], "Sincronizando ambiente (uv sync --group test)")
    if sync_code != 0:
        print("\n❌ Falha ao sincronizar o ambiente com uv. Abortando.")
        return sync_code

    test_code, output = run_pytest_and_capture(["uv", "run", "pytest", "tests/", "-v"])
    summary = parse_summary(output)
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print("RESUMO FINAL")
    print(f"{'=' * 60}")
    print(f"Testes passados : {summary['passed']}")
    print(f"Testes falhados : {summary['failed']}")
    print(f"Erros           : {summary['errors']}")
    print(f"Duração da suíte: {summary['duration']:.2f}s")
    print(f"Tempo total (com sync): {elapsed:.2f}s")

    if test_code == 0:
        print("\n✅ TODOS OS TESTES PASSARAM")
    else:
        print(f"\n❌ SUÍTE FALHOU (código de saída {test_code})")

    print(f"{'=' * 60}\n")
    return test_code


if __name__ == "__main__":
    sys.exit(main())
