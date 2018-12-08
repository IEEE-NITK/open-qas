# Retriever
This folder holds code that would perform the first part of the paper: Document retrieval.  
The main thought process and functions used are explained in this notebook [here](../notebooks/Document-Retrieval.ipynb).  
The actual module used for the pipeline is present [here](retriever.py).

The retriever is interactive. 

## Python Dependencies
 - Pandas
 - Numpy
 - sklearn
 - prettytable

All of these packages can be installed using 
```bash
    pip install [package-name]
```

## Usage
1. Get a Wikipedia JSON file. Refer [`/data`](../../data)
2. Execute the script by passing the path to the JSON File as a parameter.
```bash
    python retriever.py path/to/json/file
```
3. Pass queries to the retriever using
```python
    where_is("This is my question")
```
4. Execute `pls()` if you need any help.
5. `exit()` or ctrl+D to exit the console. 

## Results
```bash
$ python retriever.py ../../data/1000-wiki-edu-parsed.json

    Interactive Wiki Retriever
    >>> where_is(question, k=10)
    >>> pls()
    
>>> where_is("alphabet song")
+------+---------------------------+----------+
| Rank |          Document         |  Score   |
+------+---------------------------+----------+
|  1   |       Alphabet song       | 0.45752  |
|  2   |     Itsy Bitsy Spider     | 0.094688 |
|  3   |       Auld Lang Syne      | 0.086473 |
|  4   | Initial Teaching Alphabet | 0.04808  |
|  5   |          Braille          | 0.040887 |
|  6   |        Dolly Parton       | 0.038295 |
|  7   |    Imperial examination   | 0.025546 |
|  8   |      Commercium song      | 0.025288 |
|  9   |          Literacy         | 0.022508 |
|  10  |         Compendium        | 0.022232 |
+------+---------------------------+----------+
>>> 

    Exiting Interactive Wiki Retriever
```