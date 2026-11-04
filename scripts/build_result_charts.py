# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9,<4", "rapidfuzz>=3,<4"]
# ///
"""Rebuild publication charts from a compact, transcript-free evidence snapshot.

Use --refresh-data to verify and re-extract the original local evaluations.
The default works from a fresh clone, without datasets or model downloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import pathlib
import re
import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator, PercentFormatter

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "charts"
SNAPSHOT = OUT / "results.json"
LANGUAGES = ("cs", "sk")
MODELS = ("vanilla", "300h", "450h", "final")
COLORS = {
    "background": "#FAF8F4",
    "ink": "#172B36",
    "muted": "#627078",
    "grid": "#E1E3E0",
    "base": "#9BA8B0",
    "commercial": "#7862CB",
    "ours": "#007F70",
    "300h": "#7CB7AC",
    "final": "#31546D",
}


def balanced_wer(languages):
    """Equal language weight; pool word edits within each language."""
    if set(languages) != set(LANGUAGES):
        raise ValueError("Both Czech and Slovak are required")
    values = []
    for row in languages.values():
        if row["reference_words"] <= 0 or row["word_errors"] < 0:
            raise ValueError("Invalid word counts")
        values.append(row["word_errors"] / row["reference_words"])
    return sum(values) / len(values)


def validate(data):
    marker = data["commercial_comparison"]
    assert marker["reference"] == "Soniox"
    assert "soniox" not in marker["models"], "Do not score the reference against itself"
    for lang in LANGUAGES:
        counts = {m["languages"][lang]["reference_words"] for m in marker["models"].values()}
        assert len(counts) == 1, "Models must use the same reference words"
    for row in marker["models"].values():
        assert math.isfinite(balanced_wer(row["languages"]))
    cohort = data["checkpoint_comparison"]
    assert {r["reference_words"] for r in cohort["models"].values()} == {cohort["reference_words"]}
    assert len({r["speaker_seconds"] for r in cohort["models"].values()}) == 1
    for row in cohort["models"].values():
        assert row["scored"] == row["expected"] == cohort["recordings"]
        assert 0 <= row["strict_output_valid"] <= row["scored"]
        assert row["reference_words"] > 0 and row["speaker_seconds"] > 0
        assert row["word_errors"] >= 0 and row["diarization_error_seconds"] >= 0


def extract():
    """Read original artifacts, verify identities, and retain only scalar evidence."""
    from rapidfuzz.distance import Levenshtein

    hashes = {}

    def source(relative):
        path = ROOT / relative
        raw = path.read_bytes()
        hashes[str(relative)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    normalizer_path = ROOT / "experiments/periodic-eval/number_wer.py"
    spec = importlib.util.spec_from_file_location("chart_number_wer", normalizer_path)
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    hashes[str(normalizer_path.relative_to(ROOT))] = hashlib.sha256(normalizer_path.read_bytes()).hexdigest()

    base = pathlib.Path("experiments/qda-v1-release-eval")
    report = source(base / "report.json")
    selection = source(base / "selection.json")
    candidate = source(base / "v1-candidate.json")
    assert report["final"] and report["finished"] == report["expected"]
    assert report["selection_sha256"] == hashes[str(base / "selection.json")]
    assert candidate["report_sha256"] == hashes[str(base / "report.json")]
    assert candidate["checkpoint"] == "450h"

    comparison = {
        "reference": "Soniox",
        "aggregation": "Equal-weight mean of Czech and Slovak number-normalized word disagreement",
        "normalization": normalizer.VERSION,
        "recordings": 2,
        "duration_seconds": 0,
        "models": {
            "vanilla": {"label": "VibeVoice-ASR", "type": "Open-source baseline", "languages": {}},
            "450h": {"label": "ASR · ours", "type": "Open-source · QLoRA 450h", "languages": {}},
            "gemini": {"label": "Gemini 4 Argon", "type": "Commercial", "languages": {}},
        },
    }
    for lang in LANGUAGES:
        folder = pathlib.Path("experiments/production-reference-audit-v2") / lang
        reference = source(folder / "gold.reference.json")
        approval = source(folder / "gold-approval.json")
        audit = source(folder / "audit.json")
        gemini = source(folder / "gemini-independent.parsed.json")
        raw_gemini = source(folder / "gemini-independent.raw.json")
        assert raw_gemini["modelVersion"] == "gemini-4-argon"
        assert approval["reference_sha256"] == hashes[str(folder / "gold.reference.json")]
        comparison["duration_seconds"] += audit["duration_seconds"]
        ref_text = "".join(s["Content"] for s in reference)
        gem_text = " ".join(s["text"] for s in gemini["segments"])
        words = lambda text: re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
        # Reproduce the saved independent audit before applying the shared number normalizer.
        assert math.isclose(
            Levenshtein.distance(words(ref_text), words(gem_text)) / len(words(ref_text)),
            audit["independent_levenshtein_disagreement_rate"],
        )
        scored = normalizer.score(ref_text, gem_text, lang)
        comparison["models"]["gemini"]["languages"][lang] = {
            "word_errors": scored["number_normalized_errors"],
            "reference_words": scored["number_normalized_reference_words"],
        }
        for model, old_name, variant in [
            ("vanilla", "vanilla", "vanilla"),
            ("450h", "r64-full-450h", "finetuned"),
        ]:
            marker_root = pathlib.Path("experiments/qda-full-continuous-v8/markers") / old_name
            completed = source(marker_root / "completed-v3.json")
            assert hashes[str(marker_root / "completed-v3.json")] == report["markers"][model]["source_sha256"]
            assert completed["adapter_sha256"] == selection["models"][model]["adapter_sha256"]
            assert completed["provenance"][lang]["reference_sha256"] == approval["reference_sha256"]
            assert completed["provenance"][lang]["audio_sha256"] == approval["audio_sha256"]
            hypothesis = source(marker_root / variant / lang / "hypothesis.json")
            speech = [s for s in hypothesis if not re.fullmatch(r"\s*\[[^\[\]]+\]\s*", s["Content"])]
            scored = normalizer.score(ref_text, " ".join(s["Content"] for s in speech), lang)
            saved = report["markers"][model]["languages"][lang]
            for key in ("number_normalized_errors", "number_normalized_reference_words"):
                assert scored[key] == saved[key], (model, lang, key)
            comparison["models"][model]["languages"][lang] = {
                "word_errors": scored["number_normalized_errors"],
                "reference_words": scored["number_normalized_reference_words"],
            }

    rows = [r for r in selection["rows"] if r["cohort"] == "new-confirmation"]
    cohort = {
        "cohort": "new-confirmation",
        "recordings": len(rows),
        "duration_minutes": sum(r["duration"] for r in rows) / 60,
        "reference_words": report["models"]["vanilla"]["cohorts"]["new-confirmation"]["reference_words"],
        "aggregation": "Pooled word edits / reference words; pooled speaker-error seconds / scored speaker seconds",
        "models": {},
    }
    for model in MODELS:
        row = report["models"][model]["cohorts"]["new-confirmation"]
        cohort["models"][model] = {
            "word_errors": row["number_normalized_errors"],
            **{key: row[key] for key in ["reference_words", "diarization_error_seconds", "speaker_seconds", "scored", "expected", "strict_output_valid"]},
        }
        assert math.isclose(row["number_normalized_wer"], row["number_normalized_errors"] / row["reference_words"])
        assert math.isclose(row["der_250ms"], row["diarization_error_seconds"] / row["speaker_seconds"])
    data = {
        "schema_version": 1,
        "source_report_updated_at": report["updated_at"],
        "recommended_checkpoint": candidate["checkpoint"],
        "model_revision": candidate["model_revision"],
        "commercial_comparison": comparison,
        "checkpoint_comparison": cohort,
        "release_summary": {
            "main_recordings": len(selection["rows"]),
            "main_minutes": selection["main_minutes"],
            "models": {
                name: {"main": r["main"], "long": r["long"], "balanced_language_wer": r["balanced_language_wer"]}
                for name, r in report["models"].items()
            },
            "recommendation": report["recommendation"],
            "comparisons": report["comparisons"],
        },
        "methodology": [
            "Soniox machine transcripts are the reference, not an independently scored competitor.",
            "Commercial comparison uses two generated dialogues, excluded from training but repeatedly used as development markers.",
            "Gemini is the independent transcription saved before adjudication; its score is recomputed with the same normalizer as ours.",
            "The eight-recording cohort has scored outputs for all four checkpoints; it was consumed for release selection.",
            "One 450h output in that cohort is scored but not strictly valid. Scored completion and raw validity are distinct.",
            "The 450h and final models each score 32/32 main recordings and 0/2 unchunked long sessions.",
            "Lower disagreement with machine references does not establish independent human-ground-truth accuracy or a general SOTA ranking.",
        ],
        "source_sha256": dict(sorted(hashes.items())),
    }
    validate(data)
    return data


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": COLORS["ink"],
        "axes.labelcolor": COLORS["muted"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "axes.facecolor": COLORS["background"],
        "figure.facecolor": COLORS["background"],
        "savefig.facecolor": COLORS["background"],
        "svg.fonttype": "path",
        "svg.hashsalt": "asr-results-v1",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })


def page(number, title, subtitle):
    fig = plt.figure(figsize=(16, 9))
    fig.text(.055, .942, "A S R   /   R E S E A R C H", fontsize=10, weight="bold", color=COLORS["ours"])
    fig.text(.945, .942, f"{number:02d}  /  04", ha="right", fontsize=10, color=COLORS["muted"])
    fig.text(.055, .85, title, fontsize=32, weight="bold", fontfamily="DejaVu Serif")
    fig.text(.055, .798, subtitle, fontsize=13, color=COLORS["muted"])
    fig.add_artist(Line2D([.055, .945], [.765, .765], transform=fig.transFigure, lw=1, color=COLORS["grid"]))
    fig.text(.055, .043, "C Z E C H   +   S L O V A K", fontsize=9, color=COLORS["muted"])
    fig.text(.945, .043, "lofcz / asr", ha="right", fontsize=10, color=COLORS["muted"])
    return fig


def style_axes(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=COLORS["grid"], linewidth=.8)
    ax.tick_params(axis="both", length=0, pad=10)


def card(fig, x, y, width, height):
    fig.add_artist(FancyBboxPatch(
        (x, y), width, height, transform=fig.transFigure,
        boxstyle="round,pad=0.013,rounding_size=0.018", facecolor="#EDF3EF", edgecolor="none", zorder=0,
    ))


def commercial_chart(data):
    models = data["commercial_comparison"]["models"]
    fig = page(1, "Commercial & open-source ASR", "Number-normalized word disagreement with Soniox (%)  ·  Lower is better")
    ax = fig.add_axes([.265, .24, .39, .43])
    keys = ["vanilla", "450h", "gemini"]
    palette = [COLORS["base"], COLORS["ours"], COLORS["commercial"]]
    values = [100 * balanced_wer(models[k]["languages"]) for k in keys]
    ax.barh([2, 1, 0], values, height=.38, color=palette, zorder=3)
    ax.set(xlim=(0, 2.55), ylim=(-.6, 2.6), yticks=[])
    ax.xaxis.set_major_locator(MultipleLocator(.5))
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=1))
    style_axes(ax)
    for y, key, value, color in zip([2, 1, 0], keys, values, palette):
        pos = fig.transFigure.inverted().transform(ax.transData.transform((0, y)))
        fig.text(.055, pos[1] + .010, models[key]["label"], fontsize=16, weight="bold", color=color if key != "vanilla" else COLORS["ink"])
        fig.text(.055, pos[1] - .022, models[key]["type"], fontsize=10, color=COLORS["muted"])
        ax.text(value + .07, y, f"{value:.2f}%", va="center", fontsize=17, weight="bold", color=color)
    card(fig, .725, .27, .21, .37)
    reduction = 1 - values[1] / values[0]
    fig.text(.747, .567, "OURS VS. BASE", fontsize=10, weight="bold", color=COLORS["ours"])
    fig.text(.747, .455, f"{reduction:.0%}", fontsize=58, weight="bold", color=COLORS["ours"])
    fig.text(.747, .365, "fewer word\ndifferences", fontsize=19, linespacing=1.35)
    fig.text(.265, .14, "Soniox  /  commercial reference", fontsize=11, color=COLORS["muted"])
    return fig


def language_chart(data):
    models = data["commercial_comparison"]["models"]
    fig = page(2, "The language-by-language view", "Number-normalized word disagreement with Soniox (%)  ·  Lower is better")
    keys = ["vanilla", "450h", "gemini"]
    palette = [COLORS["base"], COLORS["ours"], COLORS["commercial"]]
    handles = [Line2D([], [], marker="s", linestyle="", color=c, markersize=9) for c in palette]
    fig.legend(handles, ["VibeVoice-ASR", "ASR · ours", "Gemini 4 Argon"], loc="upper left", bbox_to_anchor=(.05, .735), frameon=False, ncol=3, columnspacing=2.7)
    for left, lang, title in [(.105, "cs", "Czech"), (.57, "sk", "Slovak")]:
        ax = fig.add_axes([left, .245, .34, .36])
        values = [100 * models[k]["languages"][lang]["word_errors"] / models[k]["languages"][lang]["reference_words"] for k in keys]
        ax.bar(range(3), values, width=.44, color=palette, zorder=3)
        ax.set(ylim=(0, 2.8), xlim=(-.6, 2.6), xticks=[])
        ax.yaxis.set_major_locator(MultipleLocator(.5))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=1))
        style_axes(ax, "y")
        for x, value, color in zip(range(3), values, palette):
            ax.text(x, value + .12, f"{value:.2f}%", ha="center", fontsize=17, weight="bold", color=color)
            if value == 0:
                ax.plot(x, 0, marker="o", ms=7, color=color, clip_on=False, zorder=4)
        fig.text(left + .17, .158, title, fontsize=21, weight="bold", ha="center", fontfamily="DejaVu Serif")
    return fig


def progress_chart(data):
    rows = data["checkpoint_comparison"]["models"]
    values = [100 * rows[m]["word_errors"] / rows[m]["reference_words"] for m in MODELS]
    fig = page(3, "Fine-tuning, measured", "Number-normalized word error rate (%)  ·  Lower is better")
    ax = fig.add_axes([.105, .235, .56, .43])
    palette = [COLORS["base"], COLORS["300h"], COLORS["ours"], COLORS["final"]]
    ax.bar(range(4), values, width=.43, color=palette, zorder=3)
    ax.set(ylim=(0, 9), xlim=(-.6, 3.6), xticks=range(4), xticklabels=["VibeVoice-ASR", "Ours · 300h", "Ours · 450h", "Ours · final"])
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    style_axes(ax, "y")
    for x, value, color in zip(range(4), values, palette):
        ax.text(x, value + .3, f"{value:.2f}%", ha="center", fontsize=20, weight="bold", color=color)
    card(fig, .735, .28, .20, .35)
    reduction = 1 - values[2] / values[0]
    fig.text(.755, .562, "450h VS. BASE", fontsize=10, weight="bold", color=COLORS["ours"])
    fig.text(.755, .455, f"{100 * reduction:.1f}%", fontsize=42, weight="bold", color=COLORS["ours"])
    fig.text(.755, .362, "fewer word errors", fontsize=15)
    fig.text(.105, .13, "Reference transcripts: Soniox", fontsize=11, color=COLORS["muted"])
    return fig


def tradeoff_chart(data):
    rows = data["checkpoint_comparison"]["models"]
    fig = page(4, "Text and speaker accuracy", "Word error rate vs. diarization error rate  ·  Lower-left is better")
    ax = fig.add_axes([.115, .21, .53, .48])
    style_axes(ax, "both")
    palette = [COLORS["base"], COLORS["300h"], COLORS["ours"], COLORS["final"]]
    xs = [100 * rows[k]["word_errors"] / rows[k]["reference_words"] for k in MODELS]
    ys = [100 * rows[k]["diarization_error_seconds"] / rows[k]["speaker_seconds"] for k in MODELS]
    ax.plot(xs, ys, color="#B9C8C6", linewidth=1.8, zorder=2)
    for x, y, color, key in zip(xs, ys, palette, MODELS):
        ax.scatter(x, y, s=190 if key == "450h" else 125, color=color, edgecolor=COLORS["background"], linewidth=2, zorder=4)
    offsets = [(18, -5), (18, 3), (19, -7), (19, 7)]
    for x, y, offset, key, color in zip(xs, ys, offsets, MODELS, palette):
        label = "VibeVoice-ASR" if key == "vanilla" else "Ours · " + key
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=12, weight="bold", color=color if key != "vanilla" else COLORS["ink"])
    ax.set(xlim=(5.5, 8.4), ylim=(5.5, 15.2), xlabel="Word error rate (%)", ylabel="Diarization error rate (%)")
    ax.xaxis.set_major_locator(MultipleLocator(.5))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlabel("Word error rate (%)", labelpad=18)
    ax.set_ylabel("Diarization error rate (%)", labelpad=18)
    fig.text(.722, .623, "THE TRADEOFF", fontsize=10, weight="bold", color=COLORS["ours"])
    fig.text(.722, .53, "Better text.\nMore speaker errors.", fontsize=20, weight="bold", linespacing=1.5)
    fig.text(.722, .382, "450h checkpoint", fontsize=11, color=COLORS["muted"])
    fig.text(.722, .325, f"{xs[2]:.2f}%", fontsize=26, weight="bold", color=COLORS["ours"])
    fig.text(.84, .325, f"{ys[2]:.2f}%", fontsize=26, weight="bold", color=COLORS["ink"])
    fig.text(.722, .288, "word errors", fontsize=10, color=COLORS["muted"])
    fig.text(.84, .288, "speaker errors", fontsize=10, color=COLORS["muted"])
    return fig


def export_table(data):
    with (OUT / "values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["comparison", "model", "language", "metric", "numerator", "denominator", "percent"])
        for name, model in data["commercial_comparison"]["models"].items():
            for lang, row in model["languages"].items():
                writer.writerow(["dialogue_markers", name, lang, "number_normalized_word_disagreement", row["word_errors"], row["reference_words"], 100 * row["word_errors"] / row["reference_words"]])
            writer.writerow(["dialogue_markers", name, "balanced_cs_sk", "number_normalized_word_disagreement", "", "", 100 * balanced_wer(model["languages"])])
        for name, row in data["checkpoint_comparison"]["models"].items():
            for metric, numerator, denominator in [
                ("number_normalized_wer", "word_errors", "reference_words"),
                ("der_250ms", "diarization_error_seconds", "speaker_seconds"),
            ]:
                writer.writerow(["new_confirmation", name, "pooled_cs_sk", metric, row[numerator], row[denominator], 100 * row[numerator] / row[denominator]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true", help="Verify and extract original local artifacts")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.refresh_data:
        data = extract()
        SNAPSHOT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    else:
        data = json.loads(SNAPSHOT.read_text())
    validate(data)
    configure()
    charts = [
        ("01-commercial-open-source", commercial_chart),
        ("02-language-comparison", language_chart),
        ("03-checkpoint-progress", progress_chart),
        ("04-text-speaker-tradeoff", tradeoff_chart),
    ]
    with PdfPages(OUT / "asr-results.pdf", metadata={"Title": "ASR — Czech & Slovak results", "Author": "lofcz/asr", "CreationDate": None, "ModDate": None}) as pdf:
        for name, build in charts:
            fig = build(data)
            fig.savefig(OUT / f"{name}.png", dpi=180)
            svg = OUT / f"{name}.svg"
            fig.savefig(svg, metadata={"Date": None, "Creator": "lofcz/asr"})
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
            fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Saved {name}: PNG, SVG, PDF")
    export_table(data)
    print(f"Exported chart values and the four-page PDF; evidence snapshot records {len(data['source_sha256'])} source hashes.")


if __name__ == "__main__":
    main()
