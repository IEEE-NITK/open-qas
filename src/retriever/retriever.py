import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import prettytable
import code
import argparse
import os.path

# Array of stopwords that I gracefully borrowed from the DrQA implementation
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}

class Retriever:
    """ Once fit, will return the k best documents for a query.
    """

    def __init__(self, ngrams=2):
        """ Initialises Retriever class. 
        """
        self.h_vectorizer = HashingVectorizer(ngram_range=(1,ngrams), strip_accents='unicode', stop_words=STOPWORDS)
        self.transformer = TfidfTransformer()
        self.tfidf_matrix = None

    def build_tfidf(self, docs):
        """ Creates the tf-idf sparse matrix for the documents inputted.
            Expects a list of strings as input.
            Returns nothing.
        """
        h_doc_counts = self.h_vectorizer.transform(docs)
        self.tfidf_matrix = self.transformer.fit_transform(h_doc_counts)

    def find_best_doc_indices(self, query, k, return_scores=False):
        """ Returns an array of the indices of the k best documents.
            If return_scores=True, then also returns the respective scores 
            for every index.
            query must be a list of strings
        """
        assert self.tfidf_matrix is not None

        query_count = self.h_vectorizer.transform(query)
        query_tfidf = self.transformer.transform(query_count)
        doc_scores = np.dot(query_tfidf, np.transpose(self.tfidf_matrix))

        if(len(doc_scores.data) > k):
            ind = np.argpartition(doc_scores.data, -k)[-k:]
            ind_sort = ind[np.argsort(-doc_scores.data[ind])]
        else:
            ind_sort = np.argsort(-doc_scores.data)

        best_doc_indices = doc_scores.indices[ind_sort]

        if(not return_scores):
            return best_doc_indices
        else:
            return (best_doc_indices, doc_scores.data[ind_sort])


class WikiRetriever:
    """ Wrapper class for the actual Retriever.
        Made for the scam JSON files we're using as our _dataset_
        Modifications required before use with the actual dataset
        Pretty prints titles along with their scores.
    """

    def __init__(self, path):
        """ Initialises the WikiRetriever.
            Expects the path to the scam JSON file as input.
        """
        self.path = path

        # Load the JSON. Docs are stored in a Pandas Dataframe
        self.wikidocs = pd.read_json(path, lines=True)
        assert self.wikidocs is not None
        self.retriever = None

    def fit(self, ngrams=2):
        """ Creates the retriever and builds the tfidf matrix
            Returns nothing. 
        """
        self.retriever = Retriever(ngrams)
        self.retriever.build_tfidf(self.wikidocs.text)

    def find_best_docs(self, query, k=10):
        """ Returns the titles and scores of the k closest docs 
            query must be a list of strings
        """

        best_docs_indices, best_docs_scores = self.retriever.find_best_doc_indices(query, k, return_scores=True)
        return best_docs_indices, self.wikidocs.title[best_docs_indices], best_docs_scores

def main():
    """ This main function creates an interactive shell for the retriever.
        Heavily inspired by the DrQA implementation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to scam JSON holding Wiki data")
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