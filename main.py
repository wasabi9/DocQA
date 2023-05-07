import argparse
from docqa import DocQA


def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--api_key', type=str)

    args = parser.parse_args().__dict__
    filename = args.pop("filename")
    docqa = DocQA(file=filename)

    question = ""
    while True:
        question = input("Enter you question; enter end for ending the session: ")
        if question.lower()=="end":
            break
        print(docqa.answer_query(question)["choices"][0]["message"]["content"])

if __name__ == '__main__':
    cli()