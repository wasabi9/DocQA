import argparse
import os
from docqa import DocQA


def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--api_key', type=str)

    args = parser.parse_args().__dict__
    filename = args.pop("filename")
    os.environ['OPENAI_API_KEY'] = args.pop("api_key")
    docqa = DocQA(file=filename)

    while True:
        question = input("Enter you question; just press enter for ending the session: ")
        if question.lower() == "":
            break
        else:
            print(docqa.answer_query(question)["choices"][0]["message"]["content"])


if __name__ == '__main__':
    cli()
