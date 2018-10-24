# Datasets
This folder will hold **small, temporary datasets** that we can use to test the implementation. Actual training would be done on the 15G Compressed (50G Expanded) Wikipedia dataset. The JSON's are in Wikipedia Extractor's [document format](http://medialab.di.unipi.it/wiki/Document_Format).

## Files
 * [`1000-wiki-edu.xml`](1000-wiki-edu.xml) holds 1000 education articles in an xml with wikipedia's dump formatting.
 * [`1000-wiki-edu-parsed.json`](1000-wiki-edu-parsed.json) holds the parsed json of the above xml.
 

## Instructions to Create a new Dataset
 1. Use [PetScan](https://petscan.wmflabs.org/) to get a list of article names for a category. Use the `depth` parameter to tweak the subcategories. Save it in a csv file (or anything else) by changing the output type under the output tab.
 2. Use [Wikipedia Speacial Export](https://en.wikipedia.org/wiki/Special:Export) to get the xml file with the article content.
 3. Use [WikiExtractor](https://github.com/attardi/wikiextractor) to extract the titles and the article content from the xml file.  
 4. Push the file with the format `number_of_articles-wiki-category`. So `1000-wiki-edu` would hold roughly 1000 education articles.
