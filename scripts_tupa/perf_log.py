"""perf_log.py — Log de execução para análise de gargalos.

Cada execução de um passo gera <out_root>/logs/<passo>_<timestamp>.log com:
  * TUDO que é impresso no terminal (espelhado);
  * detalhes por edifício (tempos, janelas/linhas), indentados sob o
    arquivo/lote e sob o grupo, com separadores visuais;
  * snapshots periódicos de RECURSOS (RSS do processo, RAM disponível,
    load de CPU, disco livre, GPU se houver nvidia-smi) — para diagnosticar
    post-mortem se uma execução morreu por RAM, CPU, disco ou I/O.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from config import CFG


def _fmt_dur(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{sec:02d}s"


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


class RunLogger:
    SEP_GROUP = "═" * 78
    SEP_FILE = "─" * 78

    def __init__(self, step_name: str):
        logs_dir = CFG.out_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = logs_dir / f"{step_name}_{ts}.log"
        self._f = open(self.path, "a", encoding="utf-8")
        self.t0 = time.time()
        self.term(f"[{step_name}] log detalhado em: {self.path}")
        self.file_only(f"# início: {datetime.now().isoformat()} | "
                       f"pid={os.getpid()} | python={sys.version.split()[0]}")
        self.snapshot("início")

    # ---------------- saída ----------------
    def term(self, msg: str) -> None:
        """Terminal (minimalista) + arquivo."""
        print(msg, flush=True)
        self.file_only(msg)

    def file_only(self, msg: str) -> None:
        self._f.write(msg + "\n")
        self._f.flush()

    def group_start(self, title: str) -> None:
        self.file_only(f"\n{self.SEP_GROUP}\nGRUPO {title}  "
                       f"[{datetime.now().strftime('%H:%M:%S')}]\n{self.SEP_GROUP}")

    def group_end(self, title: str, extra: str = "") -> None:
        self.file_only(f"{self.SEP_GROUP}\nFIM DO GRUPO {title} {extra}\n"
                       f"{self.SEP_GROUP}\n")
        self.snapshot(f"fim de {title}")

    def file_start(self, title: str) -> None:
        self.file_only(f"\n  {self.SEP_FILE[:74]}\n  ARQUIVO/LOTE {title}")

    def file_end(self, title: str, extra: str = "") -> None:
        self.file_only(f"  fim {title} {extra}\n  {self.SEP_FILE[:74]}")

    def building(self, name: str, detail: str) -> None:
        """Linha por edifício, indentada sob o arquivo/lote."""
        self.file_only(f"      {name}: {detail}")

    # ---------------- recursos ----------------
    def _mem(self) -> tuple[str, str]:
        rss = avail = "?"
        try:
            with open("/proc/self/status") as f:
                for ln in f:
                    if ln.startswith("VmRSS:"):
                        rss = _fmt_bytes(int(ln.split()[1]) * 1024)
                        break
            with open("/proc/meminfo") as f:
                for ln in f:
                    if ln.startswith("MemAvailable:"):
                        avail = _fmt_bytes(int(ln.split()[1]) * 1024)
                        break
        except OSError:
            pass
        return rss, avail

    def _gpu(self) -> str:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3)
            return r.stdout.strip().replace("\n", " | ") if r.returncode == 0 else "-"
        except Exception:
            return "-"

    def snapshot(self, context: str) -> None:
        """Registra RSS/RAM/CPU/disco/GPU — a matéria-prima do diagnóstico
        de gargalo (RAM estourando, load alto, disco cheio)."""
        rss, avail = self._mem()
        try:
            load = "%.1f/%.1f/%.1f" % os.getloadavg()
        except OSError:
            load = "?"
        try:
            free = _fmt_bytes(shutil.disk_usage(CFG.out_root).free)
        except OSError:
            free = "?"
        self.file_only(f"  [recursos @ {context}] RSS={rss} | RAM_disp={avail} | "
                       f"load={load} | disco_livre={free} | gpu={self._gpu()} | "
                       f"t+{_fmt_dur(time.time() - self.t0)}")

    def close(self, summary: str = "") -> None:
        self.snapshot("fim")
        self.file_only(f"# fim: {datetime.now().isoformat()} | "
                       f"duração {_fmt_dur(time.time() - self.t0)} | {summary}")
        self._f.close()