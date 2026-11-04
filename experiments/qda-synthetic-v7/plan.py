"""Frozen 50h core + 10h monologue plan. Hours are measured again after TTS."""
import random
from core import HERE, read, save

# Minutes per language / speaker stratum: 375 = 6.25h. Eight strata = 50h.
CORE = [(0.5, 10), (1, 10), (2, 5), (5, 4), (10, 6), (20, 3), (30, 3), (60, 2)]
EXTRA = [(0.5, 10), (1, 5), (2, 5), (5, 4), (10, 5), (20, 3), (30, 3), (60, 1)]
TOPICS = [
    ('scio', 'Zkušenost s Národními srovnávacími zkouškami', 'Scio, NSZ, OSP, percentil, verbální oddíl, analytický oddíl, přijímací řízení', ['scio-main.md']),
    ('scio', 'Slovenský uchazeč a Všeobecné študijné predpoklady', 'Scio, VŠP, všeobecné študijné predpoklady, percentil, fakulta, porozumenie textu', ['vsp.md']),
    ('scio', 'Národní testování a zpětná vazba škole', 'Scio, Národní testování, čtenářská gramotnost, matematická gramotnost, výsledková zpráva', ['narodni-testovani.md']),
    ('scio', 'Mapa školy a školní klima', 'Scio, Mapa školy, dotazník, školní klima, vztahy ve třídě, anonymní odpovědi', ['mapa-skoly.md']),
    ('scio', 'Průvodci ve ScioŠkole a samostatnost dětí', 'ScioŠkola, průvodce, ScioCíle, učební cíle, reflexe, autonomie', ['scio-cile.md']),
    ('scio', 'Rodičovská zkušenost se ScioŠkolou', 'ScioŠkola, portfolio, slovní hodnocení, spolupráce, osobní odpovědnost', ['scio-cile.md']),
    ('scio', 'Emošáci a rozvoj sociálních dovedností', 'Scio, Emušáci, emoce, empatie, sociální dovednosti, porozumění', ['scio-main.md']),
    ('scio_ai', 'Příprava výuky a pracovní listy ve ScioBotu', 'ScioBot, příprava na hodinu, pracovní list, diferenciace, cíle výuky, RVP', ['sciobot.md', 'ucim-s-ai.md']),
    ('scio_ai', 'Opravování testů a formativní hodnocení', 'ScioBot, opravování testů, bodování, formativní hodnocení, zpětná vazba', ['sciobot.md']),
    ('scio_ai', 'AI chat a vlastní asistenti pro učitele', 'ScioBot, AI chat, asistent, dokumenty, prompt, školní kontext', ['sciobot.md']),
    ('scio_ai', 'Žákovské zóny, kvízy a odpovědné používání AI', 'ScioBot, žákovská zóna, interaktivní kvíz, umělá inteligence, dohled učitele', ['sciobot.md']),
    ('scio_ai', 'Učím s AI a sdílení zkušeností s kolegy', 'Učím s AI, ScioBot, digitální gramotnost, metodika, ověřování zdrojů', ['ucim-s-ai.md']),
    ('ai', 'Zavádění AI do každodenní práce učitele', 'umělá inteligence, jazykový model, prompt, kontextové okno, halucinace, ověření', ['ucim-s-ai.md']),
    ('ai', 'Přepis rozhovorů a kvalitativní výzkum', 'přepis, diarizace, časová značka, mluvčí, tematické kódování, anonymizace', []),
    ('ai', 'Vyhledávání v dokumentech a přesnost odpovědí', 'vyhledávání, embedding, vektorová databáze, citace, kontext, retrieval augmented generation', []),
    ('ai', 'Hlasové technologie v české a slovenské praxi', 'syntéza řeči, rozpoznávání řeči, akustický model, latence, token, čeština, slovenština', []),
    ('ai', 'AI projekty v malém vývojovém týmu', 'API, integrace, inference, jemné doladění, evaluační sada, nasazení, GPU', []),
    ('ai', 'Výzkumný rozhovor o změnách pracovních návyků', 'automatizace, kontrola výstupu, zpětná vazba, spolehlivost, opakovatelnost', []),
]


def plan():
    existing = read('plan.json')
    if existing:
        if existing.get('version') not in (2,3):
            raise ValueError('Unexpected prior plan; preserve it before replacing')
        extensions = read('quota-extensions.json', [])
        return {**existing, 'sessions': existing['sessions'] + extensions}
    sessions = []
    rng = random.Random(20261024)
    for cohort, mix in [('core', CORE), ('single_speaker_extra', EXTRA)]:
        for language in ('cs', 'sk'):
            for speakers in (range(1, 5) if cohort == 'core' else [1]):
                j = 0
                for minutes, count in mix:
                    for _ in range(count):
                        j += 1
                        topic = TOPICS[(j + speakers * 3 + (7 if language == 'sk' else 0)) % len(TOPICS)]
                        sid = f'{cohort}-{language}-s{speakers}-{j:03d}'
                        # Source vocabulary is chosen before script generation, never mined from answers.
                        vocab = topic[2].split(', ')
                        context = ', '.join(rng.sample(vocab, min(4, len(vocab)))) if j % 2 == 0 else ''
                        sessions.append({'id': sid, 'cohort': cohort, 'language': [language],
                            'speaker_count': speakers, 'planned_seconds': int(minutes * 60),
                            'topic_group': topic[0], 'topic': topic[1], 'vocabulary': topic[2],
                            'source_files': topic[3], 'customized_context': context,
                            'context_provenance': 'source vocabulary selected before script generation',
                            'seed': rng.randrange(2**31), 'split': 'train'})
    # Broad coverage immediately; long recordings begin early instead of waiting for all shorts.
    rng.shuffle(sessions)
    sessions.sort(key=lambda s: 0 if s['planned_seconds'] in (120, 300) else 1)
    result = {'version': 2, 'target_hours': 60, 'core_hours': 50, 'extra_single_speaker_hours': 10,
        'language_target_hours': {'cs': 30, 'sk': 30},
        'core_speaker_target_hours': {'1': 12.5, '2': 12.5, '3': 12.5, '4': 12.5},
        'duration_seconds': [30, 60, 120, 300, 600, 1200, 1800, 3600],
        'script_generator': {'engine': 'codex exec', 'model': 'gpt-6.1-sol', 'reasoning_effort': 'medium', 'interactive': False},
        'tts_model': 'sonic-3.6-2026-08-27', 'api_version': '2026-08-14',
        'script_workers': 4, 'tts_workers': 5, 'character_ceiling': 5000000,
        'credit_basis': '~1 credit per character; user-stated balance 9,075,783. Preprocessing can alter actual billing.',
        'voices': 'Native library + private clones of separate training-only speakers, distinct IDs within sessions',
        'alignment': {'precision': 'float16', 'quantization': 'int8', 'model': 'Canary auxiliary CTC'},
        'training_autostart': False, 'sessions': sessions}
    assert sum(s['planned_seconds'] for s in sessions) == 60 * 3600
    save('plan.json', result)
    return result


if __name__ == '__main__':
    from collections import Counter
    p = plan()
    print({'sessions': len(p['sessions']), 'hours': sum(s['planned_seconds'] for s in p['sessions'])/3600,
           'by_speakers': dict(Counter({n: sum(s['planned_seconds'] for s in p['sessions'] if s['speaker_count']==n)/3600 for n in range(1,5)}))})
