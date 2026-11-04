"""Match upstream rich-transcription formatting, including speakerless audio events."""
import json

EVENT_TAGS = {'[Unintelligible Speech]', '[Music]', '[Human Sounds]',
              '[Environmental Sounds]', '[Noise]', '[Silence]'}


def format_target(segments):
    result=[]
    for segment in segments:
        row={'Start':round(segment['start'],2),'End':round(segment['end'],2)}
        if segment.get('speaker') is not None:
            row['Speaker']=segment['speaker']
        row['Content']=segment['text']
        if 'Speaker' not in row and row['Content'] not in EVENT_TAGS:
            raise ValueError('Spoken text needs a speaker; unverified gaps are not event labels')
        result.append(row)
    return json.dumps(result,ensure_ascii=False,separators=(',',':'))


def context_text(row):
    values=row.get('customized_context',[])
    if not isinstance(values,list) or not all(isinstance(v,str) for v in values):
        raise ValueError('customized_context must be a list of strings')
    return '\n'.join(v.strip() for v in values if v.strip()) or None
