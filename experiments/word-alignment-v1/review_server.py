"""Local-only alignment review. Whitelisted audio routes support byte ranges."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
BENCH = HERE/'benchmarks/soniox-v1'


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(False)

    def do_HEAD(self):
        self.respond(True)

    def respond(self, head):
        path = urlsplit(self.path).path
        if path == '/':
            data, mime = (HERE/'review.html').read_bytes(), 'text/html; charset=utf-8'
        elif path == '/data':
            samples = {s['id']: json.loads((BENCH/f"{s['id']}.comparison.json").read_text())
                       for s in json.loads((BENCH/'manifest.json').read_text())}
            data, mime = json.dumps({'samples': samples}, ensure_ascii=False).encode(), 'application/json'
        elif path.startswith('/audio/'):
            allowed = {s['id']: Path(s['audio']) for s in json.loads((BENCH/'manifest.json').read_text())}
            audio = allowed.get(path.removeprefix('/audio/'))
            if audio is None:
                self.send_error(404)
                return
            size, start, end = audio.stat().st_size, 0, audio.stat().st_size-1
            requested = self.headers.get('Range')
            if requested:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
                if not match or not any(match.groups()):
                    self.send_error(416)
                    return
                first, last = match.groups()
                if first:
                    start, end = int(first), min(int(last), size-1) if last else size-1
                else:
                    start = max(0, size-int(last))
                if not 0 <= start <= end < size:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
            self.send_response(206 if requested else 200)
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end-start+1))
            if requested:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if not head:
                try:
                    with audio.open('rb') as f:
                        f.seek(start)
                        remaining = end-start+1
                        while remaining:
                            chunk = f.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if not head:
            self.wfile.write(data)


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 8781), Handler).serve_forever()
