"""Original fictional QDA scripts. No evaluation text or production interview is sent."""
import json
import re
import time
import urllib.error
import urllib.request
from core import HERE, read, save, encode, digest, env

SCRIPT_MODEL = 'gemini-3.8-flash'
TOPICS = [
 ('rekvalifikace a profesní vzdělávání', 'certifikace, kvalifikační předpoklady, příspěvek zaměstnavatele, účetnictví'),
 ('obecní služby a vyřizování žádostí', 'datová schránka, stavební povolení, katastrální mapa, správní poplatek'),
 ('péče o staršího příbuzného', 'odlehčovací služba, osobní asistence, příspěvek na péči, rehabilitace'),
 ('výběr mobilního tarifu a zákaznická podpora', 'přenos čísla, roaming, vyúčtování, reklamace'),
 ('střední škola a přechod na vysokou', 'přijímací řízení, kombinované studium, stipendium, zápočet'),
 ('každodenní dojíždění mezi obcemi', 'přestup, jízdní řád, výluka, integrovaný dopravní systém'),
 ('nákup a reklamace domácího spotřebiče', 'záruční oprava, servisní technik, sériové číslo, energetická třída'),
 ('zkušenost s pronájmem bytu', 'nájemní smlouva, vyúčtování záloh, kauce, společenství vlastníků'),
 ('návštěva ordinace a organizace péče', 'objednací lhůta, žádanka, preventivní prohlídka, ambulantní vyšetření'),
 ('zavádění podnikového informačního systému', 'přístupová oprávnění, migrace dat, dvoufaktorové ověření, integrace'),
 ('spolupráce učitelů a rodičů', 'třídní schůzka, školní poradenské pracoviště, individuální vzdělávací plán'),
 ('potraviny a plánování rodinného rozpočtu', 'jednotková cena, věrnostní program, skladování, složení výrobku'),
 ('dobrovolnictví a komunitní život', 'dobrovolnické centrum, koordinátor, veřejná sbírka, grantová žádost'),
 ('osobní finance a používání bankovnictví', 'trvalý příkaz, inkaso, úroková sazba, platební limit'),
 ('zkušenost drobného podnikatele', 'živnostenské oprávnění, fakturace, splatnost, zálohy na pojištění'),
 ('použitelnost digitální veřejné služby', 'bankovní identita, elektronický podpis, formulář, přístupnost'),
 ('bydlení a energetická renovace', 'tepelná izolace, fotovoltaika, distribuční sazba, energetický posudek'),
 ('kulturní akce a volnočasové aktivity', 'abonentní vstupenka, předprodej, kulturní centrum, rezervační systém'),
 ('začleňování nového zaměstnance', 'adaptační plán, mentor, zkušební doba, bezpečnostní školení'),
 ('život lidí s pohybovým omezením', 'bezbariérový přístup, kompenzační pomůcka, doprovod, orientační systém'),
 ('malý výrobní podnik a kontrola kvality', 'výrobní šarže, kalibrace, toleranční pole, reklamační protokol'),
 ('zkušenost se zahraničním pobytem', 'uznávání kvalifikace, pobytové oprávnění, nostrifikace, zdravotní pojištění'),
 ('výzkumná práce a analýza rozhovorů', 'polostrukturovaný rozhovor, anonymizace, tematické kódování, výzkumná otázka'),
 ('organizace směn v sociálních službách', 'rozpis směn, předávací dokumentace, supervize, krizová intervence'),
 ('založení a provoz malé kavárny', 'hygienické požadavky, dodavatel, provozní náklady, sezónnost'),
 ('sportovní klub a práce s mládeží', 'členský příspěvek, tréninkový plán, soustředění, fyzioterapie'),
 ('zemědělství a lokální distribuce', 'osevní postup, závlaha, farmářský trh, odběratelská smlouva'),
 ('logistika a provoz e-shopu', 'expedice, skladová evidence, vratka, dobírka, výdejní místo'),
 ('celoživotní vzdělávání v knihovně', 'digitální gramotnost, katalog, meziknihovní výpůjčka, komunitní dílna'),
 ('dlouhodobá zkušenost s komunitní péčí', 'terénní služba, koordinace, individuální plán, dostupnost služeb'),
 ('proměna pracovního postupu ve vývojovém týmu', 'revize kódu, testovací prostředí, nasazení, zpětná vazba, technický dluh'),
]


def plan():
    existing = read('plan.json')
    if existing:
        return existing
    durations = [10] * 18 + [20] * 6 + [30] * 4 + [60] * 3
    sessions = []
    cs = json.loads((HERE / 'research/voices-cs.json').read_text())['data']
    sk = json.loads((HERE / 'research/voices-sk.json').read_text())['data']
    for i, (minutes, topic) in enumerate(zip(durations, TOPICS)):
        language = 'sk' if i % 3 == 2 else 'cs'
        voices = cs if language == 'cs' else sk
        female = [v for v in voices if v['gender'] == 'feminine']
        male = [v for v in voices if v['gender'] == 'masculine']
        pair = [female[(i // 3) % len(female)], male[(i // 2) % len(male)]]
        sessions.append({'id': f'{language}-{i+1:03d}', 'language': language,
            'topic': topic[0], 'vocabulary': topic[1], 'planned_minutes': minutes,
            'parts': minutes // 10, 'speakers': [{'id': n, 'voice_id': v['id'],
                'voice_name': v['name'], 'gender': v['gender'], 'speed': [0.96, 1.0, 1.04][(i+n) % 3]}
                for n, v in enumerate(pair)]})
    result = {'version': 1, 'status': 'preparation', 'target_hours': 10,
              'script_model': SCRIPT_MODEL, 'tts_model': 'sonic-3.6-2026-08-27',
              'concurrent_tts_requests': 2, 'tts_character_ceiling': 650000,
              'tts_usd_per_1000_characters_estimate': 0.05,
              'cost_note': 'Configured estimate, actual account pricing may differ; no subscription changes.',
              'sessions': sessions, 'source': 'Original fictional scenarios; no evaluation transcripts used',
              'real_train_hours': 473.21705322916665, 'training_autostart': False}
    save('plan.json', result)
    return result


def make_part(session, part, target_words=1450):
    relative = f'scripts/{session["id"]}/part-{part:02d}'
    old = read(relative + '.json')
    if old:
        return old
    previous = read(f'scripts/{session["id"]}/part-{part-1:02d}.json', {})
    schema = {'type': 'object', 'properties': {
        'continuity_notes': {'type': 'string', 'description': 'Names, personal details and events to keep consistent in the next chapter.'},
        'turns': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'speaker': {'type': 'integer', 'enum': [0, 1]},
            'text': {'type': 'string', 'description': 'Exact natural spoken words, with punctuation and native diacritics; no stage directions.'}},
            'required': ['speaker', 'text'], 'additionalProperties': False}}},
        'required': ['continuity_notes', 'turns'], 'additionalProperties': False}
    prompt = f'''Create an ORIGINAL fictional qualitative research interview in {'CZECH' if session['language']=='cs' else 'SLOVAK'}.
Topic: {session['topic']}. Useful domain vocabulary (translate naturally into the interview language): {session['vocabulary']}.
This is part {part+1} of {session['parts']} of ONE coherent interview, scenario {session['id']}.
Speaker 0 is a female qualitative researcher; speaker 1 is a male participant describing his own experiences.
Write about {target_words} spoken words, 18-32 turns. The respondent talks much more than the researcher.
Use follow-up questions grounded in preceding answers, concrete anecdotes, mixed short and long turns, occasional natural hesitations, corrections and uncertainty. Do not produce an essay, list, vocabulary lesson, or factual expert lecture. Vary syntax and pacing. Avoid repetitive generic openings or summaries.
Include a few plausible fictional names and places, dates, quantities, times and inflected forms of domain vocabulary, but do not cram jargon into every turn. Write numbers as the exact words to be spoken in this language, in their grammatically appropriate forms. Keep Czech and Slovak distinct, retaining every accent. Never put speaker names, role labels, markup, abbreviations intended as stage directions, or bracketed non-speech tags in the spoken text. Do not add fictional laughter or silence labels.
No advice is needed: these are fictional experiences, including for health or money topics.
Only introduce the interview in the first part. Only close it in the last part. Each turn should be below 1800 characters; split a very long answer into successive turns by the same speaker if necessary.
Continuity from earlier parts: {previous.get('continuity_notes','No preceding part.')}
Last preceding turns: {json.dumps(previous.get('turns',[])[-4:],ensure_ascii=False)}
Return JSON matching the supplied schema.'''
    body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'generationConfig': {'responseMimeType': 'application/json', 'responseJsonSchema': schema,
                                 'maxOutputTokens': 16000, 'temperature': 1.0}}
    intent = read(relative + '.intent.json')
    raw = read(relative + '.response.json')
    if intent and not raw:
        raise RuntimeError('Prior script request is ambiguous; inspect before resubmitting: ' + relative)
    if not raw:
        save(relative + '.request.json', body)
        save(relative + '.intent.json', {'submitted_at': time.time(), 'request_sha256': digest(encode(body)), 'model': SCRIPT_MODEL})
        request = urllib.request.Request(f'https://generativelanguage.googleapis.com/v1beta/models/{SCRIPT_MODEL}:generateContent',
            data=encode(body), headers={'x-goog-api-key': env()['GOOGLE_API_KEY'], 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                raw = json.load(response)
            save(relative + '.response.json', raw)
        except urllib.error.HTTPError as e:
            save(relative + '.error.json', {'status': e.code, 'message': e.read(3000).decode(errors='replace')})
            raise RuntimeError('Script API HTTP ' + str(e.code)) from None
    candidate = raw['candidates'][0]
    if candidate.get('finishReason') != 'STOP':
        raise ValueError('Incomplete script output: ' + candidate.get('finishReason', 'unknown'))
    text = ''.join(p.get('text', '') for p in candidate['content']['parts'] if not p.get('thought'))
    parsed = json.loads(text)
    turns = parsed['turns']
    if not turns or set(t['speaker'] for t in turns) != {0, 1}:
        raise ValueError('Missing interview speakers')
    for turn in turns:
        if not isinstance(turn['text'], str) or not turn['text'].strip() or len(turn['text']) > 4000:
            raise ValueError('Invalid utterance length')
        if re.search(r'[\[\]<>]', turn['text']):
            raise ValueError('Markup in spoken transcript')
    parsed.update({'session_id': session['id'], 'part': part, 'language': session['language'],
                   'words': sum(len(t['text'].split()) for t in turns),
                   'script_model_version': raw.get('modelVersion'), 'usage': raw.get('usageMetadata'),
                   'source': 'new_fictional_generation', 'response_sha256': digest(encode(raw))})
    if parsed['words'] < 700:
        raise ValueError('Script unexpectedly short; retain response for inspection')
    save(relative + '.json', parsed)
    return parsed
