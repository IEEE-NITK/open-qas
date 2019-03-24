import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
from openqas.retriever.retriever import WikiRetriever
import prettytable
import code
from deeppavlov import build_model, configs
model = build_model(configs.squad.squad, download=True)

def main():
    """
    This main function creates an interactive shell for the retriever.
    Heavily inspired by the DrQA implementation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to Wiki database")
    parser.add_argument('-l', '--load', type=str, help='path to existing tfidf matrix')
    args = parser.parse_args()

    assert os.path.isfile(args.path)
    ranker = WikiRetriever(args.path)

    if(args.load is not None):
        ranker.load_ids()
        ranker.load(args.load)
    else:
        print("Loading Docs")
        ranker.load_docs()
        print("Starting TFIDF calcaulation")
        ranker.fit()
        print("Saving matrix")
        ranker.save(args.path + '.tfidf.pkl')

    def where_is(query, k=10):
        doc_ids, doc_titles, doc_scores,docs = ranker.find_best_docs([query], k, return_docs=True)
        ptable = prettytable.PrettyTable(
            ['Rank', 'Doc ID', 'Title', 'Score']
        )
        for i in range(len(doc_ids)):
            ptable.add_row([i+1, doc_ids[i], doc_titles[i], '%.5g' % doc_scores[i]])

        print(ptable)
        
        answer = docs[0]
        score = -1
        print(docs)
        for context in docs:
            answer_array = model([context],[quest])
            ans_score =  answer_array[2][0]
            if(ans_score > score ):
                answer = answer_array[0][0]
                score = ans_score

        print(answer)

    banner = """
    Interactive Wiki Retriever
    >>> where_is(question, k=10)
    >>> pls()
    """

    exitmsg = """
    Exiting Interactive Wiki Retriever
    """

    def pls():
        print(banner)
    code.interact(banner=banner, local=locals(), exitmsg=exitmsg)

if __name__ == "__main__":
    main()
