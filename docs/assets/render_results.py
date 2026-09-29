"""Plot the README's selected benchmarks from docs/RESULTS.md."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ASSETS = Path(__file__).resolve().parent


def read_table(text: str, header: str) -> list[tuple[str, float]]:
    """Read method names and FAD-CLAP from one named Markdown table."""
    lines = text.split(header, 1)[1].splitlines()[2:]
    values = []
    for line in lines:
        if not line.startswith("|"):
            break
        cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
        values.append((cells[0], float(cells[2])))
    if len(values) != 6 or sum(name == "Meric" for name, _ in values) != 1:
        raise ValueError(f"Expected Meric and five baselines under {header}")
    return values


def main() -> None:
    results = (ASSETS.parent / "RESULTS.md").read_text()
    panels = [
        ("Vision to music", "MelBench", "| Method | MelBench FAD-MERT", 1.08, 0.2),
        ("Text to music", "MusicCaps", "| Method | MusicCaps FAD-MERT", 0.49, 0.1),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.axisbelow": True})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
    fig.set_facecolor("white")
    accent, neutral, ink = "#147d77", "#aab6c2", "#243342"
    for ax, (task, dataset, header, limit, spacing) in zip(axes, panels, strict=True):
        rows = sorted(read_table(results, header), key=lambda row: row[1])
        labels = [name.replace(" + ", " +\n") for name, _ in rows]
        values = [value for _, value in rows]
        colors = [accent if name == "Meric" else neutral for name, _ in rows]
        ax.barh(range(len(rows)), values, color=colors, height=0.56)
        ax.set_yticks(range(len(rows)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, limit)
        ax.xaxis.set_major_locator(MultipleLocator(spacing))
        ax.set_xlabel("FAD-CLAP (lower is better)", color=ink, labelpad=12)
        ax.set_title(f"{task}\n{dataset}", loc="left", fontsize=17, weight="bold", color=ink, pad=20)
        ax.grid(axis="x", color="#e6eaee", linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#c6ced6")
        ax.tick_params(axis="y", length=0, pad=10, colors=ink)
        ax.tick_params(axis="x", colors="#566573")
        for index, (name, value) in enumerate(rows):
            ax.text(
                value + limit * 0.025,
                index,
                f"{value:.3f}",
                va="center",
                color=ink,
                weight="bold" if name == "Meric" else "normal",
            )
        for tick, (name, _) in zip(ax.get_yticklabels(), rows, strict=True):
            if name == "Meric":
                tick.set_color(accent)
                tick.set_weight("bold")
    fig.savefig(ASSETS / "generation-results.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
