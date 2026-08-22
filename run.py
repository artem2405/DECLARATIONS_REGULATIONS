import argparse

from extract_declarations import extract_declarations
from make_vectors import make_vectors
from rerank_regulations import rerank_regulations
from retrieve_context import retrieve_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default="./out")
    args = parser.parse_args()

    make_vectors(args.data)
    extract_declarations(args.data)
    retrieve_context(args.data)
    rerank_regulations(args.data, args.out)


if __name__ == "__main__":
    main()
