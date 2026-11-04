import os
os.environ.setdefault('HF_HOME','/data/work/asr-pilot/hf')
from huggingface_hub import snapshot_download
MODEL_REVISION='d0c9efdb8d614685062c04425d91e01b6f37d944'
TOKENIZER_REVISION='d149729398750b98c0af14eb82c78cfe92750796'
def model_paths():
    model=snapshot_download('microsoft/VibeVoice-ASR',revision=MODEL_REVISION,cache_dir=os.path.join(os.environ['HF_HOME'],'hub'))
    tokenizer=snapshot_download('Qwen/Qwen2.5-7B',revision=TOKENIZER_REVISION,cache_dir=os.path.join(os.environ['HF_HOME'],'hub'),allow_patterns=['*token*','*.json'])
    return model,tokenizer
