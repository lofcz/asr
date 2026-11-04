"""Stop both collectors at the duration target or conservative storage cutoff."""
import datetime
import json
import pathlib
import subprocess

from audio_worker import atomic_json, storage_ready


def stop_reason(hours, bytes_used, config):
    if bytes_used >= config['budget_stop_bytes']:
        return 'storage_cutoff'
    if hours >= config['target_hours']:
        return 'duration_target_reached'
    return None


def main():
    state = pathlib.Path.home() / '.local/state/asr-downloader'
    config = json.loads((state / 'config.json').read_text())
    if not storage_ready(config):
        return
    root = pathlib.Path(config['root'])
    # Refresh before making the stop decision, rather than trusting a stale report.
    subprocess.run(['/usr/bin/python', str(pathlib.Path(__file__).with_name('corpus_status.py'))],
                   check=True, stdout=subprocess.DEVNULL, timeout=60)
    size = int(subprocess.check_output(['du', '-sb', str(root)], timeout=30).split()[0])
    report = json.loads((root / 'data/manifests/corpus-status.json').read_text())
    hours = report['downloaded_hours_after_known_duplicate_exclusions']
    reason = stop_reason(hours, size, config)
    status = {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'downloaded_hours': hours, 'project_bytes_including_partial_files': size,
              'target_hours': config['target_hours'], 'budget_bytes': config['budget_bytes'],
              'budget_stop_bytes': config['budget_stop_bytes'], 'stop_reason': reason}
    if reason:
        # Disable too: reaching the limit must survive a reboot. Queues are retained.
        subprocess.run(['systemctl', '--user', 'disable', '--now',
                        'asr-audio-worker.service', 'asr-rss-batch.service'], check=True, timeout=110)
    atomic_json(state / 'collection-guard.json', status)
    atomic_json(root / 'data/manifests/collection-guard.json', status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
