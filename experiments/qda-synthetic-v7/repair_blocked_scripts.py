"""Repair rejected, unpaid drafts while the ordinary audio queue keeps running."""
import json
import re
import time
from core import DATA, read, save
from generate_scripts import make_part, REPAIR_POLICY


def main():
    tasks = []
    for state_path in sorted((DATA/'sessions').glob('*/state.json')):
        state = json.loads(state_path.read_text())
        match = re.search(r'Script validation attempts exhausted; retained: scripts-v4/([^/]+)/part-(\d+)', state.get('error', ''))
        if state.get('status') == 'error' and match:
            tasks.append((match[1], int(match[2])))
    report = {'repair_policy': REPAIR_POLICY, 'started_at': time.time(),
              'requested_parts': len(tasks), 'results': [], 'paid_tts_requests': 0}
    save('vocabulary/script-repair-status.json', report)
    for sid, part in tasks:
        try:
            session = read(f'sessions/{sid}/spec.json')
            contracts = {int(p.stem.split('-')[1]): json.loads(p.read_text())
                         for p in (DATA/'vocabulary/contracts'/sid).glob('part-*.json')}
            elapsed = 0
            total_words = 0
            previous = None
            for i in range(part):
                completed = read(f'sessions/{sid}/parts/{i:03d}/complete.json')
                if not completed:
                    raise ValueError('Earlier part is not committed')
                previous = read(completed['script_path'])
                elapsed += completed['duration_seconds']
                total_words += previous['words']
            remaining = max(20, session['planned_seconds']-elapsed)
            weight = sum(c['planned_seconds'] for i, c in contracts.items() if i >= part)
            seconds = max(20, remaining*contracts[part]['planned_seconds']/weight)
            rate = min(190, max(110, total_words/elapsed*60)) if elapsed else 145
            result = make_part(session, part, seconds, part == max(contracts), previous, rate)
            entry = {'session': sid, 'part': part, 'status': 'script_ready',
                     'words': result['words'], 'target_words': result['target_words'],
                     'script_path': result['script_path']}
        except Exception as e:
            entry = {'session': sid, 'part': part, 'status': 'error', 'error': str(e)}
        report['results'].append(entry)
        report['updated_at'] = time.time()
        save('vocabulary/script-repair-status.json', report)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    report['finished_at'] = time.time()
    save('vocabulary/script-repair-status.json', report)


if __name__ == '__main__':
    main()
