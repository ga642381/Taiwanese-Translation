import os
import subprocess
import requests
import json
from subprocess import PIPE, STDOUT
import pathlib


class TranslationAPI():
    def __init__(self, model):
        self.model = model
        self.this_path = pathlib.Path(__file__).parent.absolute()
        self.model_dir = os.path.join(self.this_path, 'model')
        self.start_allen_server()
        
    def start_allen_server(self):
        if self.model == 'transformer':
            model_name = 'bert_transformer'
            model_path = os.path.join(self.model_dir, f'{model_name}.tar.gz')
        
        port = 8000
        port_using = self.check_port_usage(port)
        if port_using:
            print(f"port {port} is using...")
        else:
            print(f"Starting Allennlp server on port {port}")
            cmd = f'allennlp serve --archive-path {model_path} --predictor seq2seq --field-name source'
            print(cmd)
            self.proc = subprocess.Popen(cmd, shell=True)
            #self.proc = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    
    def check_port_usage(self, port):
        cmd = f"fuser {port}/tcp"
        p = subprocess.run(cmd, capture_output=True, shell=True)
        if p.stdout:
            return True
        else:
            return False

    def translate(self, txt):
        url = "http://127.0.0.1:8000/predict"
        req_data = {"source":txt}
        req_headers = {'Content-Type':'application/json'}
        
        rsp = requests.post(url, headers=req_headers, json=req_data)
        translated_tokens =  eval(rsp.content.decode())['predicted_tokens']
        return ' '.join(translated_tokens)

    def shut_down_server(self):
        self.proc.kill()
        
if __name__ == '__main__':
    API = TranslationAPI('transformer')
    API.start_allen_server()
    
    
    
    # print(proc.stdout.read().split('\n')[-2])
    # OSError: [Errno 98] Address already in use: ('127.0.0.1', 8000)
    # if running, no return
