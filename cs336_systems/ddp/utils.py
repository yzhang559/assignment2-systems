import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class DDPCommRow:
    backend: str
    device: str
    world_size: int
    op: str
    size_bytes: int
    dtype: str
    warmup_steps: int
    measure_steps: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    algbw_GBps: float
    busbw_GBps: float


class DDPCommBenchmarkReporter:
    """
    DDP communication benchmark reporter:
      - append rows to JSONL
      - render markdown table
      - write markdown to file
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        md_path: str | Path,
        *,
        title: str = "#### DDP communication benchmark (single node)",
        float_fmt: str = ".3f",
        sort_cols: Optional[List[str]] = None,
        cols: Optional[List[str]] = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.md_path = Path(md_path)
        self.title = title
        self.float_fmt = float_fmt

        self.sort_cols = sort_cols or ["backend", "device", "world_size", "op", "size_bytes", "dtype"]
        self.cols = cols or [
            "backend",
            "device",
            "world_size",
            "op",
            "size_bytes",
            "dtype",
            "warmup_steps",
            "measure_steps",
            "mean_ms",
            "std_ms",
            "min_ms",
            "max_ms",
            "algbw_GBps",
            "busbw_GBps",
        ]

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: DDPCommRow) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    def read_df(self) -> pd.DataFrame:
        if not self.jsonl_path.exists():
            return pd.DataFrame()
        records: List[Dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _center_align_markdown(md: str) -> str:
        lines = md.splitlines()
        if len(lines) < 2:
            return md
        header = lines[0]
        n_cols = header.count("|") - 1
        centered = "| " + " | ".join([":---:" for _ in range(n_cols)]) + " |"
        return "\n".join([lines[0], centered] + lines[2:])

    def render_markdown(self) -> str:
        df = self.read_df()
        if df.empty:
            return f"{self.title}\n\n(no rows)\n"

        df = df.copy()
        df = df.sort_values(self.sort_cols)
        df = df[self.cols]

        for c in ["mean_ms", "std_ms", "min_ms", "max_ms", "algbw_GBps", "busbw_GBps"]:
            if c in df.columns:
                df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{float(x):{self.float_fmt}}")

        md = [self.title, "", self._center_align_markdown(df.to_markdown(index=False)), ""]
        return "\n".join(md)

    def write_markdown(self) -> None:
        self.md_path.write_text(self.render_markdown(), encoding="utf-8")

    def plot_results(self, output_dir: str | Path = ".") -> None:
        df = self.read_df()
        if df.empty:
            print("No data to plot.")
            return

        df = df.copy()
        df["size_MB"] = df["size_bytes"] / 1024**2
        # average duplicate runs for the same config
        agg = (
            df.groupby(["backend", "world_size", "size_MB"], as_index=False)
            .agg(busbw_GBps=("busbw_GBps", "mean"), mean_ms=("mean_ms", "mean"))
        )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # --- Plot 1: busbw vs. tensor size, one line per world_size ---
        fig, ax = plt.subplots(figsize=(7, 5))
        for ws in sorted(agg["world_size"].unique()):
            sub = agg[agg["world_size"] == ws].sort_values("size_MB")
            ax.plot(sub["size_MB"], sub["busbw_GBps"], marker="o", label=f"world_size={ws}")
        ax.set_xscale("log")
        ax.set_xlabel("Tensor size (MB)")
        ax.set_ylabel("Bus bandwidth (GB/s)")
        ax.set_title("All-reduce busbw vs. tensor size")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        fig.tight_layout()
        p1 = out / "allreduce_busbw_vs_size.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {p1}")

        # --- Plot 2: busbw vs. world_size, one line per tensor size ---
        fig, ax = plt.subplots(figsize=(7, 5))
        for size_mb in sorted(agg["size_MB"].unique()):
            sub = agg[agg["size_MB"] == size_mb].sort_values("world_size")
            label = f"{size_mb:.0f} MB" if size_mb >= 1 else f"{size_mb*1024:.0f} KB"
            ax.plot(sub["world_size"], sub["busbw_GBps"], marker="o", label=label)
        ax.set_xlabel("Number of processes (world_size)")
        ax.set_ylabel("Bus bandwidth (GB/s)")
        ax.set_title("All-reduce busbw vs. number of processes")
        ax.set_xticks(sorted(agg["world_size"].unique()))
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        p2 = out / "allreduce_busbw_vs_worldsize.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {p2}")