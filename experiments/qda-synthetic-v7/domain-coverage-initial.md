Audited **76 accepted script parts in 76 sessions**, containing **812 spoken turns** and **36,062 words**. There are 38 Czech and 38 Slovak sessions. Snapshot: 2026-10-24T00:13:28.192613+00:00.

Only `turns[*].text` in files matching `^part-\d+\.json$` was counted. Attempts, schemas, intents, continuity notes, metadata, prompts and research excerpts are excluded from term totals. Prompts and production source were read separately to inspect the target-text policy.

**ScioLink is absent. NSZ, OSP and VŠP have no canonical written acronym occurrences.** NSZ and VŠP appear as expanded names; OSP appears only as `ó es pé`.

| Requested term | Occurrences | Sessions | Czech occurrences | Slovak occurrences | Observed surface variants (count) |
|---|---:|---:|---:|---:|---|
| ScioBot | 36 | 21 | 17 | 19 | `ScioBot` (11); `ScioBota` (10); `ScioBote` (3); `ScioBotem` (5); `ScioBotom` (4); `ScioBotu` (3) |
| ScioŠkola | 5 | 5 | 2 | 3 | `ScioŠkole` (5) |
| ScioCíle | 4 | 3 | 2 | 2 | `ScioCieľmi` (1); `ScioCíle` (2); `ScioCílům` (1) |
| ScioLink | 0 | 0 | 0 | 0 | — |
| Scio (samostatný název) | 24 | 24 | 12 | 12 | `Scio` (23); `Sciu` (1) |
| NSZ | 3 | 3 | 2 | 1 | `národné porovnávacie skúšky` (1); `Národní srovnávací zkoušky` (2) |
| OSP | 3 | 3 | 2 | 1 | `ó es pé` (3) |
| VŠP | 6 | 6 | 3 | 3 | `všeobecné študijné predpoklady` (4); `všeobecných študijných predpokladov` (2) |
| Mapa školy | 9 | 5 | 4 | 5 | `Mapa školy` (2); `Mape školy` (1); `Mapou školy` (2); `Mapu školy` (2); `Mapě školy` (2) |
| Národní testování | 6 | 6 | 3 | 3 | `Národní testování` (5); `Národního testování` (1) |
| Emušáci | 10 | 4 | 5 | 5 | `Emušáci` (5); `Emušákoch` (1); `Emušákov` (2); `Emušáky` (2) |

Specific Scio product names and their inflections are mutually exclusive from generic Scio: `ScioBotem`, `ScioŠkole` and `ScioCieľmi` do not create additional generic Scio hits. Variant-specific session counts, per-language session counts and every occurrence are included in the JSON report.

| Acronym family | Written acronym | Spoken-letter spelling | Full expansion |
|---|---:|---:|---:|
| NSZ | 0 | 0 | 3 |
| OSP | 0 | 3 | 0 |
| VŠP | 0 | 0 | 6 |
| AI / umělá či umelá inteligence | 5 | 2 | 19 |
| LLM | 0 | 0 | 0 |
| RAG | 0 | 0 | 4 |
| ASR / rozpoznávání řeči | 0 | 0 | 5 |
| TTS / syntéza řeči | 0 | 0 | 5 |
| API | 0 | 3 | 0 |
| GPU | 0 | 1 | 0 |

**All 76 saved prompts explicitly instruct the generator to write initialisms as spoken letter names.** Their hashes match the accepted scripts. This has produced **9 phonetic acronym occurrences in 8 sessions**, including one `gé pé ú` that a search limited to OSP/AI/API would miss.

| Current target text | Canonical target | Occurrences | Sessions |
|---|---|---:|---:|
| `ej aj` | `AI` | 2 | 2 |
| `ej pí aj` | `API` | 1 | 1 |
| `gé pé ú` | `GPU` | 1 | 1 |
| `á pé í` | `API` | 2 | 2 |
| `ó es pé` | `OSP` | 3 | 3 |

The 76 saved accepted prompts contain the historical instruction on line 7: “Spell initialisms as spoken letter names where needed (e.g. ó es pé), or use the full term.” **The live generator has been corrected during this audit:** `generate_scripts.py:85` now requires canonical written forms and explicitly prohibits phonetic letter names in transcripts. The accepted scripts and their saved prompts still retain the historical text. `pipeline.py:77` sends the same `turn['text']` to TTS, and `pipeline.py:128,130` copies it into segment `Content`; `align_dataset.py:45` also uses the stored text. The JSON includes exact source lines and hashes.

Of the 9 accepted-script phonetic occurrences, **9** are also present in currently available assembled `labels.json` segment text. No audio was listened to or altered. The source flow makes the remaining occurrences a target-policy issue even where assembled labels are not yet present.

Use canonical written acronyms in ASR targets and keep pronunciation aids separate if the speech synthesizer needs them. Fully expanded names such as `Národní srovnávací zkoušky` are valid ordinary text; they are not the same error as phonetic letter-name targets. Natural inflected brand forms should remain grammatical.

Additional observed issues:

- `ScioCieľmi` occurs once. It is grouped with ScioCíle for coverage but is flagged as a translated brand stem under a prompt that asks to preserve product names.
- `plan.py` uses `Emošáci` in a topic title while its vocabulary and accepted spoken text use `Emušáci`. This is a prompt-source typo, not an accepted spoken variant.

AI coverage below is deliberately split by specificity. API/GPU/evaluation sets are technical vocabulary that can occur outside AI; chat/asistent/automatizace are even broader. These counts are lexical coverage and should not be presented as a semantic count of AI discussions.

| AI or related term | Category | Occurrences | Sessions | Observed surface variants (count) |
|---|---|---:|---:|---|
| AI / umělá či umelá inteligence | explicit_ai_or_speech | 26 | 20 | `AI` (5); `ej aj` (2); `umelej inteligencie` (1); `umelej inteligencii` (1); `umelou inteligenciou` (5); `umelá inteligencia` (4); `umelú inteligenciu` (1); `umělou inteligencí` (2); `umělá inteligence` (4); `umělé inteligenci` (1) |
| jazykový model | explicit_ai_or_speech | 5 | 5 | `jazykovému modelu` (1); `Jazykový model` (3); `jazykový model` (1) |
| prompt | explicit_ai_or_speech | 12 | 10 | `Prompt` (1); `prompt` (8); `promptu` (3) |
| kontextové okno | explicit_ai_or_speech | 5 | 5 | `Kontextové okno` (1); `kontextové okno` (4) |
| halucinace / halucinácia | explicit_ai_or_speech | 5 | 5 | `halucinace` (3); `halucinácia` (1); `halucinácie` (1) |
| embedding | explicit_ai_or_speech | 5 | 4 | `Embedding` (3); `embedding` (2) |
| vektorová databáze / databáza | explicit_ai_or_speech | 5 | 4 | `vektorovou databázou` (1); `vektorová databáza` (1); `Vektorová databáze` (1); `vektorová databáze` (2) |
| RAG | explicit_ai_or_speech | 4 | 4 | `retrieval augmented generation` (4) |
| inference / inferencia | explicit_ai_or_speech | 3 | 3 | `Inference` (1); `inferenci` (1); `inferenciou` (1) |
| jemné doladění / doladenie | explicit_ai_or_speech | 3 | 3 | `jemné doladenie` (1); `Jemné doladění` (1); `jemné doladění` (1) |
| token | explicit_ai_or_speech | 5 | 5 | `token` (5) |
| ASR / rozpoznávání řeči | explicit_ai_or_speech | 5 | 5 | `rozpoznávania reči` (1); `rozpoznávanie reči` (1); `Rozpoznávání řeči` (1); `rozpoznávání řeči` (2) |
| TTS / syntéza řeči | explicit_ai_or_speech | 5 | 5 | `Syntéza řeči` (2); `syntéza řeči` (1); `syntéze reči` (2) |
| akustický model | explicit_ai_or_speech | 5 | 5 | `akustický model` (5) |
| diarizace / diarizácia | explicit_ai_or_speech | 4 | 4 | `Diarizace` (2); `Diarizácia` (2) |
| API | ai_associated_technical | 3 | 3 | `ej pí aj` (1); `á pé í` (2) |
| GPU | ai_associated_technical | 1 | 1 | `gé pé ú` (1) |
| evaluační / evaluačná sada | ai_associated_technical | 3 | 3 | `evaluační sada` (2); `evaluačnú sadu` (1) |
| automatizace / automatizácia | contextual_low_specificity | 5 | 5 | `Automatizace` (1); `automatizace` (2); `automatizácia` (1); `automatizácie` (1) |
| chat / čet | contextual_low_specificity | 9 | 6 | `chat` (3); `chatu` (3); `Čet` (1); `čet` (2) |
| asistent | contextual_low_specificity | 19 | 5 | `Asistent` (2); `asistent` (10); `asistenta` (4); `asistentov` (1); `asistentovi` (2) |

Other scanned AI terms with zero matches: LLM, chatbot, ChatGPT, Claude, Gemini, strojové učení / učenie, neuronová / neurónová síť. These are audit coverage gaps, not a claim that every term must appear in every session.

The additional project `Učím s AI` appears 4 times in 4 sessions. Its `AI` substring is intentionally included in the AI view, so product and AI totals must not be added together.

Explicit AI/speech vocabulary reaches 33 sessions; including associated technical vocabulary reaches 33. Low-specificity vocabulary raises this to 38, but is not evidence by itself that a session discusses AI.

This is an initial local snapshot, not coverage of all planned hours. Term presence does not establish pronunciation, audio completion, factual accuracy or balanced representation. Regexes, all observed variants, exact evidence offsets, input hashes and saved-prompt hashes are preserved in `domain-coverage-initial.json`. Only these two audit artifacts were written by this audit; source changes made concurrently by the parent agent are distinguished above.
