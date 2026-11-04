"""Current dashboard at the established :8766 URL; historic evaluation URLs survive."""
import http.client
import http.server
import urllib.parse
import pathlib
import mimetypes

CURRENT = {'/', '/training', '/training-data', '/preparation', '/preparation-data',
           '/experiments', '/probes', '/calibration', '/review', '/api', '/status', '/acoustic-review',
           '/markers', '/marker-comparisons', '/marker-audio'}
HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te',
       'trailer', 'transfer-encoding', 'upgrade', 'content-length'}


class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def forward(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == '/release-evaluation':
            self.send_response(302)
            self.send_header('Location', '/release-evaluation/')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if path.startswith('/release-evaluation/'):
            name = path.removeprefix('/release-evaluation/') or 'index.html'
            allowed = {'index.html','report.json','report.md','status.json','word-errors.csv','v1-candidate.json'}
            if self.command not in ('GET','HEAD') or name not in allowed:
                self.send_error(404);return
            file = pathlib.Path(__file__).resolve().parent.parent/'qda-v1-release-eval'/name
            if not file.is_file():
                self.send_error(404, 'Evaluation artifact is not available yet');return
            data = file.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', (mimetypes.guess_type(name)[0] or 'application/octet-stream')+'; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            if self.command != 'HEAD':self.wfile.write(data)
            return
        current = path in CURRENT or path == '/audio'
        target = self.path
        if path == '/legacy':
            current, target = False, '/'
        upstream = http.client.HTTPConnection('127.0.0.1', 8767 if current else 8768, timeout=60)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP | {'host'}}
        # The existing review endpoint validates same-origin write requests.
        if current and headers.get('Origin') in ('http://localhost:8766', 'http://127.0.0.1:8766'):
            headers['Origin'] = 'http://localhost:8767'
        body = self.rfile.read(int(self.headers.get('Content-Length', 0))) if self.command == 'POST' else None
        sent = False
        try:
            upstream.request(self.command, target, body=body, headers=headers)
            response = upstream.getresponse()
            self.send_response(response.status)
            length = response.getheader('Content-Length')
            for k, v in response.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            if length is not None:
                self.send_header('Content-Length', length)
            else:
                self.send_header('Connection', 'close')
                self.close_connection = True
            self.end_headers()
            sent = True
            if self.command != 'HEAD':
                while chunk := response.read(65536):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            if not sent:
                self.send_error(502, 'Dashboard upstream unavailable')
        finally:
            upstream.close()

    do_GET = do_HEAD = do_POST = forward


if __name__ == '__main__':
    http.server.ThreadingHTTPServer(('127.0.0.1', 8766), Proxy).serve_forever()
