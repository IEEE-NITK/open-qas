import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import openqas.utils.db
from openqas.utils.db import WikiDB

def test():
    wiki = WikiDB()
    print(wiki.get_doc_text('43568'))
    # print(len(wiki.get_all_doc_ids()))
    wiki.close()

test()