"""Verify portable raw inputs and the exact authorized transcript targets."""
import json, hashlib, math, time
import torch
from runtime import HERE, paths, prepare_item, atomic, sha, TOKENIZER_REVISION
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def main():
    torch.set_num_threads(1)
    base, tok = paths()
    processor = VibeVoiceASRProcessor.from_pretrained(base, language_model_pretrained_name=tok)
    manifest = json.loads((HERE/'manifest.json').read_text())
    ready = json.loads((HERE/'inputs-ready.json').read_text())
    assert ready['manifest_sha256'] == sha(HERE/'manifest.json')
    train, dev = manifest['train'], manifest['development']
    assert not ({r['record_id'] for r in train} & {r['record_id'] for r in dev})
    rows = train + dev
    assert len({r['window_id'] for r in rows}) == len(rows)
    assert {x['window_id'] for x in ready['inputs']} == {r['window_id'] for r in rows}
    totals = {'train': 0, 'development': 0}
    for row in rows:
        item = prepare_item(row, processor)
        receipt = json.loads((HERE/'raw-inputs'/(row['window_id']+'.json')).read_text())
        assert receipt['source_audio_sha256'] == row['audio_sha256']
        assert receipt['source_start'] == row['source_start'] and receipt['source_end'] == row['source_end']
        assert receipt['learned_features_cached'] is False
        assert torch.isfinite(item['speech']).all() and item['speech'].ndim == 1
        assert item['window_id'] == row['window_id']
        assert item['mask'].dtype == torch.bool and int(item['mask'].sum()) == math.ceil(item['speech'].numel()/3200)
        body = json.dumps(row['target'], ensure_ascii=False, separators=(',', ':'))
        expected = processor.tokenizer.apply_chat_template([{'role':'assistant','content':body}], tokenize=True, add_generation_prompt=False)
        assert len(expected) == item['target_tokens']
        n = len(item['ids']) - len(expected)
        assert n > 0 and item['ids'][n:].tolist() == expected and item['labels'][n:].tolist() == expected
        assert (item['labels'][:n] == -100).all() and not item['mask'][n:].any()
        for segment in row['target']:
            assert 0 <= segment['Start'] <= segment['End'] <= row['duration'] + .02
            assert isinstance(segment['Content'], str) and segment['Content'].strip()
        totals[row['split']] += 1
    result = {'status':'passed', 'manifest_sha256':sha(HERE/'manifest.json'),
              'inputs_ready_sha256':sha(HERE/'inputs-ready.json'), 'rows_checked':totals,
              'transcript_tokens_exact':True, 'train_dev_source_overlap':0,
              'waveform_and_masks_checked':True, 'learned_features_cached':False,
              'soniox_labels_changed':False, 'checked_at':time.time()}
    atomic(HERE/'input-verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__': main()
