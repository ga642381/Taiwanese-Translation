import requests
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from 臺灣言語工具.解析整理.拆文分析器 import 拆文分析器
from 臺灣言語工具.語音合成 import 台灣話口語講法

# AllenNLP 翻譯
AllenNLP_url = "http://127.0.0.1:8000/predict"


def translate(txt):
    # 華語
    #txt = "今天天氣真好"
    req_data = {"source": txt}
    req_headers = {'Content-Type': 'application/json'}
    rsp = requests.post(AllenNLP_url, headers=req_headers, json=req_data)
    # ['kin1', 'a2', 'jit8', 'thinn1', 'khi3', 'tsin1', 'ho2']
    translated_tokens = eval(rsp.content.decode())['predicted_tokens']

    # 'kin1 a2 jit8 thinn1 khi3 tsin1 ho2'
    pinyin = ' '.join(translated_tokens)
    return pinyin


if __name__ == '__main__':
    """example usage
    python translate.py sentence "今天天氣真好"
    python translate.py file test.txt
    """
    parser = argparse.ArgumentParser(description='翻譯')
    subparsers = parser.add_subparsers(
        help='help for subcommand', dest="subcommand")
    # === subcommand: sentence ===
    parser_sentence = subparsers.add_parser('sentence', help='翻譯句子')
    parser_sentence.add_argument('sentence', help='翻譯的句子')
    # === subcommand: file ===
    parser_file = subparsers.add_parser('file', help='翻譯文件')
    parser_file.add_argument('file_in', help='翻譯的文件')
    args = parser.parse_args()

    if args.subcommand == 'sentence':
        pinyin = translate(args.sentence)
        print(pinyin)

    elif args.subcommand == 'file':
        # === read file === #
        with open(args.file_in, 'r') as f:
            txt = f.readlines()

        # === translate === #
        result = []
        for i, t in enumerate(tqdm(txt)):
            data = {}
            text = t.strip()
            pinyin = translate(t)
            data["index"] = i
            data["華語"] = text
            data["臺語"] = pinyin
            result.append(data)
        # === save file === #
        with open(Path(args.file_in).with_suffix(".json"), 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        assert False, "unknown subcommand"
