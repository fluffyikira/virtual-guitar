from http.server import HTTPServer, BaseHTTPRequestHandler
from sys import argv
import json
from playsound import playsound
import winsound

BIND_HOST = 'localhost'
PORT = 8080

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.write_response(b'')

    def do_POST(self):
        content_length = int(self.headers.get('content-length', 0))
        body = self.rfile.read(content_length)
        self.write_response(body)

    def write_response(self, content):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(content)

        raw = content.decode('utf-8')
        data = json.loads(raw)
        #print(data)
        #handVal = data["hand"]
        chordVal = data["value"]
        print(chordVal)
        #print(handVal)
        #playsound("chord_sounds/{}.wav".format(chordVal))
        print("Playing the {} chord sound".format(chordVal))
        winsound.PlaySound("chord_sounds/{}_slow.wav".format(chordVal),winsound.SND_FILENAME)

if len(argv) > 1:
    arg = argv[1].split(':')
    BIND_HOST = arg[0]
    PORT = int(arg[1])

print(f'Listening on http://{BIND_HOST}:{PORT}\n')

httpd = HTTPServer((BIND_HOST, PORT), SimpleHTTPRequestHandler)
httpd.serve_forever()