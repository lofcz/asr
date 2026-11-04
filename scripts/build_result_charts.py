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
from matplotlib.ticker import MaxNLocator

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "charts"
SNAPSHOT = OUT / "results.json"
LANGUAGES = ("cs", "sk")
MODELS = ("vanilla", "300h", "450h", "final")
COLORS = {
    "vanilla": "#999999", "gemini": "#AA4499", "whisper": "#D55E00",
    "450h": "#0072B2", "300h": "#56B4E9", "final": "#009E73",
}
LABELS = {
    "vanilla": "VibeVoice-ASR", "gemini": "Gemini 4 Argon",
    "whisper": "Whisper large-v3-turbo", "450h": "Ours (450h)",
    "300h": "Ours (300h)", "final": "Ours (final)",
}
HATCHES = {"vanilla": "", "gemini": "..", "whisper": "//", "450h": "", "300h": "\\", "final": "xx"}



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
    for row in cohort.get("baselines", {}).values():
        assert row["scored"] == row["expected"] == cohort["recordings"]
        assert row["reference_words"] == cohort["reference_words"]
        assert row["word_errors"] >= 0
        assert "diarization_error_seconds" not in row, "Do not invent diarization for a text-only baseline"


def attach_whisper(data, run, expected):
    """Admit only the complete run on the exact audio/reference pairs."""
    assert run["complete"] and run["scored"] == run["expected"] == len(expected)
    rows = run["rows"]
    assert len(rows) == len(expected) and {r["id"] for r in rows} == set(expected)
    assert run["settings"]["model"] == "openai/whisper-large-v3-turbo"
    markers = {}
    cohort = {"word_errors": 0, "reference_words": 0, "scored": 0, "expected": data["checkpoint_comparison"]["recordings"]}
    for row in rows:
        source = expected[row["id"]]
        assert row["complete"]
        for key in ["cohort", "language", "audio_sha256", "reference_sha256", "source_start"]:
            assert row[key] == source[key], (row["id"], key)
        assert abs(row["duration_seconds"] - source["duration_seconds"]) < .001
        m = row["metrics"]
        assert m["number_normalized_reference_words"] == source["reference_words"]
        assert math.isclose(m["number_normalized_wer"], m["number_normalized_errors"] / m["number_normalized_reference_words"])
        counts = {"word_errors": m["number_normalized_errors"], "reference_words": m["number_normalized_reference_words"]}
        if row["cohort"] == "dialogue_markers":
            assert row["language"] not in markers
            markers[row["language"]] = counts
        else:
            assert row["cohort"] == "new-confirmation"
            cohort["word_errors"] += counts["word_errors"]
            cohort["reference_words"] += counts["reference_words"]
            cohort["scored"] += 1
    assert set(markers) == set(LANGUAGES)
    data["commercial_comparison"]["models"]["whisper"] = {
        "label": LABELS["whisper"], "type": "Open-source baseline", "languages": markers,
    }
    data["checkpoint_comparison"]["baselines"] = {"whisper": cohort}
    data["whisper_benchmark"] = run
    validate(data)


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
        "aggregation": "Macro-averaged WER across Czech and Slovak",
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
        "schema_version": 3,
        "wer_definition": {
            "name": "Word error rate", "abbreviation": "WER", "unit": "percent",
            "formula": "100 * (substitutions + deletions + insertions) / reference_tokens",
            "alignment": "Minimum unit-cost Levenshtein alignment of preprocessed word-token sequences",
            "text_preprocessing": normalizer.VERSION,
            "reference_length": "Number of reference tokens after applying the same preprocessing to reference and hypothesis",
            "marker_aggregation": "Arithmetic mean of the Czech and Slovak WERs (macro-average)",
            "cohort_aggregation": "Sum of per-recording edit counts divided by sum of reference-token counts (corpus WER)",
        },
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
    whisper_folder = pathlib.Path("experiments/whisper-large-v3-turbo")
    if (ROOT / whisper_folder / "result.json").exists():
        run = source(whisper_folder / "result.json")
        assert run["settings"]["selection_sha256"] == hashes[str(base / "selection.json")]
        assert run["settings"]["normalizer_sha256"] == hashes[str(normalizer_path.relative_to(ROOT))]
        assert run["settings"]["runner_sha256"] == hashlib.sha256((ROOT / "scripts/benchmark_whisper.py").read_bytes()).hexdigest()
        expected = {}
        reference_texts = {}
        for lang in LANGUAGES:
            folder = pathlib.Path("experiments/production-reference-audit-v2") / lang
            approval = source(folder / "gold-approval.json")
            audit = source(folder / "audit.json")
            key = "marker-" + lang
            expected[key] = {
                "cohort": "dialogue_markers", "language": lang, "audio_sha256": approval["audio_sha256"],
                "reference_sha256": approval["reference_sha256"], "source_start": 0,
                "duration_seconds": audit["duration_seconds"],
                "reference_words": comparison["models"]["vanilla"]["languages"][lang]["reference_words"],
            }
            reference_texts[key] = "".join(s["Content"] for s in source(folder / "gold.reference.json"))
        events = {"[Unintelligible Speech]", "[Music]", "[Human Sounds]", "[Environmental Sounds]", "[Noise]", "[Silence]"}
        for row in rows:
            reference_text = " ".join(s["Content"] for s in row["target"] if s["Content"].strip() not in events)
            expected[row["window_id"]] = {
                "cohort": "new-confirmation", "language": row["language"], "audio_sha256": row["audio_sha256"],
                "reference_sha256": hashlib.sha256((json.dumps(row["target"], ensure_ascii=False, indent=2) + "\n").encode()).hexdigest(),
                "source_start": row["source_start"], "duration_seconds": row["duration"],
                "reference_words": len(normalizer.tokens(reference_text, row["language"])),
            }
            reference_texts[row["window_id"]] = reference_text
        for result in run["rows"]:
            raw = source(whisper_folder / (result["id"] + ".json"))
            assert {k: v for k, v in raw.items() if k != "hypothesis"} == result
            assert normalizer.score(reference_texts[result["id"]], raw["hypothesis"], result["language"]) == result["metrics"]
        attach_whisper(data, run, expected)
        data["methodology"].append("Whisper large-v3-turbo uses pinned official weights, FP16, beam size 5, known language, no prompt, no VAD or silence skipping; no speaker labels or DER are assigned.")
        data["source_sha256"] = dict(sorted(hashes.items()))
    validate(data)
    return data


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 10,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.titlesize": 11,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "text.color": "black",
        "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": True, "axes.spines.bottom": True,
        "axes.linewidth": .65, "grid.linewidth": .5,
        "svg.fonttype": "path", "svg.hashsalt": "asr-paper-results-v2", "pdf.fonttype": 42,
        "hatch.linewidth": .4,
    })


def style_axes(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="0.88")
    ax.tick_params(direction="out", length=3, width=.6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]))
    if axis != "x":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]))


def marker_keys(data):
    return [key for key in ["gemini", "whisper", "vanilla", "450h"]
            if key in data["commercial_comparison"]["models"]]


def commercial_chart(data):
    models = data["commercial_comparison"]["models"]
    keys = marker_keys(data)
    values = [100 * balanced_wer(models[k]["languages"]) for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 2.8), layout="constrained")
    for y, key, value in zip(range(len(keys)), keys, values):
        ax.barh(y, value, height=.52, color=COLORS[key], edgecolor="0.25", linewidth=.45, hatch=HATCHES[key])
        ax.text(value + max(values) * .025, y, f"{value:.2f}", va="center", fontsize=10)
    ax.set(yticks=range(len(keys)), yticklabels=[LABELS[k] for k in keys],
           xlabel="Macro-averaged WER (%)", xlim=(0, max(values) * 1.18))
    ax.invert_yaxis()
    style_axes(ax, "x")
    return fig


def language_chart(data):
    models = data["commercial_comparison"]["models"]
    keys = marker_keys(data)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True, layout="constrained")
    ymax = max(100 * models[k]["languages"][lang]["word_errors"] /
               models[k]["languages"][lang]["reference_words"] for k in keys for lang in LANGUAGES)
    for ax, lang, title in zip(axes, LANGUAGES, ["(a) Czech", "(b) Slovak"]):
        for x, key in enumerate(keys):
            row = models[key]["languages"][lang]
            value = 100 * row["word_errors"] / row["reference_words"]
            ax.bar(x, value, width=.58, color=COLORS[key], edgecolor="0.25", linewidth=.45, hatch=HATCHES[key])
            ax.text(x, value + ymax * .025, f"{value:.2f}", ha="center", fontsize=9)
        style_axes(ax)
        ax.set(xticks=range(len(keys)), xticklabels=[{"gemini": "Gemini", "whisper": "Whisper", "vanilla": "VibeVoice", "450h": "Ours"}[k] for k in keys],
               ylim=(0, ymax * 1.2), title=title)
    axes[0].set_ylabel("Word error rate (%)")
    return fig


def progress_chart(data):
    cohort = data["checkpoint_comparison"]
    rows = {**cohort.get("baselines", {}), **cohort["models"]}
    keys = (["whisper"] if "whisper" in rows else []) + list(MODELS)
    values = [100 * rows[k]["word_errors"] / rows[k]["reference_words"] for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), layout="constrained")
    for x, key, value in zip(range(len(keys)), keys, values):
        ax.bar(x, value, width=.6, color=COLORS[key], edgecolor="0.25", linewidth=.45, hatch=HATCHES[key])
        ax.text(x, value + max(values) * .025, f"{value:.2f}", ha="center", fontsize=10)
    style_axes(ax)
    ax.set(xticks=range(len(keys)), xticklabels=[LABELS[k].replace("Whisper large-v3-turbo", "Whisper\nlarge-v3-turbo").replace("VibeVoice-ASR", "VibeVoice\nASR") for k in keys],
           ylim=(0, max(values) * 1.2), ylabel="Corpus WER (%)")
    return fig


def tradeoff_chart(data):
    rows = data["checkpoint_comparison"]["models"]
    fig, ax = plt.subplots(figsize=(5.4, 3.5), layout="constrained")
    xs = [100 * rows[k]["word_errors"] / rows[k]["reference_words"] for k in MODELS]
    ys = [100 * rows[k]["diarization_error_seconds"] / rows[k]["speaker_seconds"] for k in MODELS]
    ax.plot(xs, ys, color="0.65", lw=.8, zorder=2)
    for x, y, key, marker, offset in zip(xs, ys, MODELS, ["o", "^", "s", "D"], [(7, -4), (7, 1), (7, -4), (7, 0)]):
        ax.scatter(x, y, s=35, marker=marker, color=COLORS[key], edgecolor="0.25", linewidth=.45, zorder=3)
        ax.annotate(LABELS[key], (x, y), xytext=offset, textcoords="offset points", fontsize=9, va="center")
    style_axes(ax, "both")
    ax.set(xlim=(5.7, 8.4), ylim=(6.2, 14.5), xlabel="Corpus WER (%)",
           ylabel="Diarization error rate (%)")
    return fig


def export_table(data):
    with (OUT / "values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["comparison", "model", "language", "metric", "numerator", "denominator", "percent", "aggregation", "text_preprocessing"])
        for name, model in data["commercial_comparison"]["models"].items():
            for lang, row in model["languages"].items():
                writer.writerow(["dialogue_markers", name, lang, "wer", row["word_errors"], row["reference_words"], 100 * row["word_errors"] / row["reference_words"], "per_language", data["commercial_comparison"]["normalization"]])
            writer.writerow(["dialogue_markers", name, "balanced_cs_sk", "wer", "", "", 100 * balanced_wer(model["languages"]), "macro_by_language", data["commercial_comparison"]["normalization"]])
        for name, row in data["checkpoint_comparison"]["models"].items():
            for metric, numerator, denominator in [
                ("wer", "word_errors", "reference_words"),
                ("der_250ms", "diarization_error_seconds", "speaker_seconds"),
            ]:
                writer.writerow(["new_confirmation", name, "pooled_cs_sk", metric, row[numerator], row[denominator], 100 * row[numerator] / row[denominator], "corpus" if metric == "wer" else "pooled_speaker_time", data["commercial_comparison"]["normalization"] if metric == "wer" else ""])
        for name, row in data["checkpoint_comparison"].get("baselines", {}).items():
            writer.writerow(["new_confirmation", name, "pooled_cs_sk", "wer", row["word_errors"], row["reference_words"], 100 * row["word_errors"] / row["reference_words"], "corpus", data["commercial_comparison"]["normalization"]])


def update_readme_tables(data):
    path = ROOT / "README.md"
    if not path.exists():
        return
    text = path.read_text()
    model_lines = ["| Model | Czech WER (%) | Slovak WER (%) | Macro-averaged WER (%) |", "|---|---:|---:|---:|"]
    for key in marker_keys(data):
        row = data["commercial_comparison"]["models"][key]
        values = [100 * row["languages"][lang]["word_errors"] / row["languages"][lang]["reference_words"] for lang in LANGUAGES]
        model_lines.append(f"| {LABELS[key]} | {values[0]:.2f} | {values[1]:.2f} | {100 * balanced_wer(row['languages']):.2f} |")
    checkpoint_lines = ["| Model | Corpus WER (%) | DER (%) | Scored recordings |", "|---|---:|---:|---:|"]
    cohort = data["checkpoint_comparison"]
    rows = {**cohort.get("baselines", {}), **cohort["models"]}
    for key in (["whisper"] if "whisper" in rows else []) + list(MODELS):
        row = rows[key]
        wer = 100 * row["word_errors"] / row["reference_words"]
        der = f"{100 * row['diarization_error_seconds'] / row['speaker_seconds']:.2f}" if "diarization_error_seconds" in row else "—"
        checkpoint_lines.append(f"| {LABELS[key]} | {wer:.2f} | {der} | {row['scored']}/{row['expected']} |")
    for tag, lines in [("MODEL", model_lines), ("CHECKPOINT", checkpoint_lines)]:
        start, end = f"<!-- {tag}_TABLE_START -->", f"<!-- {tag}_TABLE_END -->"
        if start in text and end in text:
            before, rest = text.split(start, 1)
            _, after = rest.split(end, 1)
            text = before + start + "\n" + "\n".join(lines) + "\n" + end + after
    path.write_text(text)


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
            fig.savefig(OUT / f"{name}.png", dpi=300)
            svg = OUT / f"{name}.svg"
            fig.savefig(svg, metadata={"Date": None, "Creator": "lofcz/asr"})
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
            fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Saved {name}: PNG, SVG, PDF")
    export_table(data)
    update_readme_tables(data)
    print(f"Exported chart values and the four-page PDF; evidence snapshot records {len(data['source_sha256'])} source hashes.")


if __name__ == "__main__":
    main()
