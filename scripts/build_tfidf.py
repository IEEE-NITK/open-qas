import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import openqas.utils.db
from openqas.utils.db import WikiDB

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
    ranker.fit()

    def where_is(query, k=10):
        doc_indices, doc_titles, doc_scores = ranker.find_best_docs([query], k)
        ptable = prettytable.PrettyTable(
            ['Rank', 'Document', 'Score']
        )
        for i in range(len(doc_indices)):
            ptable.add_row([i+1, doc_titles[doc_indices[i]], '%.5g' % doc_scores[i]])
        
        print(ptable)

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