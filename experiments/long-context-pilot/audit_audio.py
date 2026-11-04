"""Independent VAD/energy audit. Gaps are candidates, never automatically 'silence'."""
import hashlib, importlib.metadata, json, pathlib, subprocess, time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from transcript_utils import merge_words

BASE = pathlib.Path(__file__).resolve().parent


def union(intervals, duration):
    result = []
    for start, end in sorted((max(0., a), min(duration, b)) for a,b in intervals if b>a):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def complement(intervals, duration):
    result, pos = [], 0.
    for start, end in union(intervals, duration):
        if start > pos:
            result.append([pos,start])
        pos=end
    if pos < duration:
        result.append([pos,duration])
    return result


def overlap(a, intervals):
    return sum(max(0., min(a[1], b[1])-max(a[0], b[0])) for b in intervals)


def main():
    torch.set_num_threads(1)
    model=load_silero_vad()
    out=BASE/'annotation-audit';out.mkdir(exist_ok=True)
    summary=[]
    for source in json.loads((BASE/'sample/sources.json').read_text()):
        started=time.monotonic()
        audio=BASE/'sample'/source['id']/'audio.flac'
        wave,sample_rate=sf.read(audio,dtype='float32')
        assert sample_rate == 24000 and wave.ndim == 1
        wave=resample_poly(wave,2,3).astype(np.float32)
        duration=len(wave)/16000
        raw=json.loads((audio.parent/'soniox.json').read_text())['tokens']
        words=merge_words(raw)
        speech=get_speech_timestamps(torch.from_numpy(wave),model,sampling_rate=16000,
                                    threshold=.5,min_speech_duration_ms=250,min_silence_duration_ms=100,
                                    speech_pad_ms=30,return_seconds=False)
        vad=[[r['start']/16000,r['end']/16000] for r in speech]
        word_regions=union([[w['start_ms']/1000-.1,w['end_ms']/1000+.1] for w in words],duration)
        # Preserve disagreement as a separate review signal; don't suppress labelled speech.
        candidates=[]
        for start,end in complement(vad+word_regions,duration):
            if end-start < .5:
                continue
            samples=wave[int(start*16000):int(end*16000)]
            frames=samples[:len(samples)//320*320].reshape(-1,320)
            rms=np.sqrt(np.mean(frames.astype(np.float64)**2,axis=1))
            rms_db=20*np.log10(np.maximum(rms,1e-12))
            peak_db=float(20*np.log10(max(float(np.abs(samples).max()),1e-12)))
            very_quiet=float(np.percentile(rms_db,95)) < -55 and peak_db < -40
            candidates.append({'start':round(start,4),'end':round(end,4),'duration':round(end-start,4),
                               'rms_dbfs_p95':round(float(np.percentile(rms_db,95)),2),
                               'peak_dbfs':round(peak_db,2),'candidate_tag':'[Silence]' if very_quiet else None,
                               'status':'needs_acoustic_verification',
                               'reason':'low_energy_and_no_speech_evidence' if very_quiet else 'no_vad_or_soniox_speech_evidence'})
        untranscribed_vad=[]
        for interval in vad:
            covered=overlap(interval,word_regions)
            if interval[1]-interval[0]>=.5 and covered/(interval[1]-interval[0])<.25:
                untranscribed_vad.append({'start':interval[0],'end':interval[1],'word_coverage':covered,
                                         'status':'review_missing_speech_or_vad_false_positive'})
        tokens_without_speaker=[t for t in raw if not t.get('is_audio_event') and t.get('speaker') is None]
        result={'source_id':source['id'],'split':source['split'],'duration':duration,
                'audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest(),
                'silero_version':importlib.metadata.version('silero-vad'),
                'vad_config':{'threshold':.5,'sampling_rate':16000,'min_speech_duration_ms':250,'min_silence_duration_ms':100,'speech_pad_ms':30},
                'vad_speech':vad,'vad_speech_fraction':sum(b-a for a,b in vad)/duration,
                'candidate_non_speech':candidates,'untranscribed_vad':untranscribed_vad,
                'raw_soniox_events':[t for t in raw if t.get('is_audio_event')],
                'tokens_without_speaker':tokens_without_speaker,
                'policy':'VAD negatives and ASR gaps are not silence labels. No candidate is silently promoted to training truth.',
                'elapsed_seconds':time.monotonic()-started}
        (out/f"{source['id']}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
        row={k:result[k] for k in ('source_id','split','duration','vad_speech_fraction','elapsed_seconds')}
        row.update(non_speech_candidates=len(candidates),candidate_seconds=sum(c['duration'] for c in candidates),
                   possible_missing_speech_regions=len(untranscribed_vad),soniox_events=len(result['raw_soniox_events']))
        summary.append(row);print(json.dumps(row),flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
