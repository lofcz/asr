"""Differentiable VibeVoice encoders, including exact causal context at tile edges.

Calls encoder modules directly: upstream encode()/encode_speech() disable autograd.
Only raw, processor-normalized waveform may be cached. No learned features are cached.
"""
import math
import torch
from torch.utils.checkpoint import checkpoint

SAMPLE_RATE = 24000


def convolution(layer):
    value = layer
    while not isinstance(value, torch.nn.Conv1d):
        if not hasattr(value, 'conv'):
            return None
        value = value.conv
    return value


def receptive_field(encoder):
    """Finite causal receptive field in input samples; no attention in these encoders."""
    span, stride = 1, 1
    def add(layer):
        nonlocal span, stride
        conv = convolution(layer)
        if conv is None:
            return
        if conv.padding != (0,):
            raise ValueError('Unexpected internal convolution padding')
        span += (conv.kernel_size[0] - 1) * conv.dilation[0] * stride
        stride *= conv.stride[0]
    if not encoder.causal:
        raise ValueError('Overlap encoder path requires causal convolution')
    for down, blocks in zip(encoder.downsample_layers, encoder.stages):
        for layer in down:
            add(layer)
        for block in blocks:
            if not hasattr(block.mixer, 'conv'):
                raise ValueError('Unknown temporal mixer; recompute the receptive field')
            add(block.mixer.conv)
    add(encoder.head)
    return span, stride


def encode_means(encoder, waveform, tile_seconds=30., checkpoint_activations=True):
    # Waveform is B x 1 x T. Tile starts and overlap are aligned to output stride.
    span, stride = receptive_field(encoder)
    tile = max(stride, math.floor(tile_seconds * SAMPLE_RATE / stride) * stride)
    overlap = math.ceil((span - 1) / stride) * stride
    outputs = []
    for start in range(0, waveform.shape[-1], tile):
        begin = max(0, start - overlap)
        stop = min(waveform.shape[-1], start + tile)
        piece = waveform[..., begin:stop]
        def encode(x):
            return encoder(x, use_cache=False)
        out = checkpoint(encode, piece, use_reentrant=False) if checkpoint_activations and torch.is_grad_enabled() else encode(piece)
        discard = (start - begin) // stride
        keep = math.ceil((stop - start) / stride)
        outputs.append(out[..., discard:discard + keep])
    means = torch.cat(outputs, dim=-1).transpose(1, 2).contiguous()
    if means.shape[1] != math.ceil(waveform.shape[-1] / stride):
        raise ValueError('Encoder tile alignment changed feature count')
    return means


def encode_speech_trainable(model, speech, tile_seconds=30., checkpoint_activations=True):
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerEncoderOutput
    base = model.model
    if speech.ndim == 2:
        speech = speech.unsqueeze(1)
    speech = speech.to(device=next(base.acoustic_tokenizer.encoder.parameters()).device,
                       dtype=next(base.acoustic_tokenizer.encoder.parameters()).dtype)
    acoustic = encode_means(base.acoustic_tokenizer.encoder, speech, tile_seconds, checkpoint_activations)
    semantic = encode_means(base.semantic_tokenizer.encoder, speech, tile_seconds, checkpoint_activations)
    distribution = VibeVoiceTokenizerEncoderOutput(mean=acoustic, std=base.acoustic_tokenizer.fix_std)
    acoustic_tokens = distribution.sample(dist_type=base.acoustic_tokenizer.std_dist_type)[0]
    features = base.acoustic_connector(acoustic_tokens) + base.semantic_connector(semantic)
    audio_modules = (base.acoustic_tokenizer.encoder, base.semantic_tokenizer.encoder,
                     base.acoustic_connector, base.semantic_connector)
    if torch.is_grad_enabled() and any(p.requires_grad for module in audio_modules for p in module.parameters()) and not features.requires_grad:
        raise RuntimeError('Speech path is detached from autograd')
    return features


def trainable_groups(model):
    groups = {'acoustic_encoder': [], 'semantic_encoder': [], 'acoustic_connector': [],
              'semantic_connector': [], 'language_model': [], 'lm_head': []}
    prefixes = {'model.acoustic_tokenizer.encoder.': 'acoustic_encoder',
                'model.semantic_tokenizer.encoder.': 'semantic_encoder',
                'model.acoustic_connector.': 'acoustic_connector',
                'model.semantic_connector.': 'semantic_connector',
                'model.language_model.': 'language_model', 'lm_head.': 'lm_head'}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [group for prefix, group in prefixes.items() if name.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError('Unclassified trainable parameter: ' + name)
        groups[matches[0]].append((name, parameter))
    if not all(groups.values()):
        raise ValueError('A required encoder, connector, decoder or output head is frozen/missing')
    return groups
