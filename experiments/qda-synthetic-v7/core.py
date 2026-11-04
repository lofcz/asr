"""Durable, credential-free artifacts on the SSD and an internal-disk replica."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'datasets/cartesia-qda-v7'
REPLICA = Path('/home/lofcz/.local/share/asr-synthetic/cartesia-qda-v7')
ENV = ROOT.parent / 'tazatelka/.env'


def env():
    result = {}
    for line in ENV.read_text().splitlines():
        key, sep, value = line.partition('=')
        if sep and not key.lstrip().startswith('#'):
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def put(relative, data):
    if not (ROOT / '.asr-volume-id').exists():
        raise OSError('Expected dataset SSD is unavailable')
    atomic(REPLICA / relative, data)
    atomic(DATA / relative, data)


def save(relative, value):
    put(relative, encode(value))


def read(relative, default=None):
    path = DATA / relative
    return json.loads(path.read_bytes()) if path.exists() else default


def verified(relative, expected):
    for base in (DATA, REPLICA):
        path = base / relative
        if path.exists() and sha(path) == expected:
            data = path.read_bytes()
            for other in (DATA, REPLICA):
                dest = other / relative
                if not dest.exists() or sha(dest) != expected:
                    atomic(dest, data)
            return data
    raise ValueError('No valid replica for ' + str(relative))
