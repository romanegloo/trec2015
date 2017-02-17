File Structure
==============
- bin/, execution files, shell scripts 
    - trec_setup.sh: initial setup
    - trec_terrier.sh: main controller
    - interactive_terrier.sh: query
- var/, indexing and retrieval results

Indexing
========
1. let indexing know which directory to search for document pool
    'bin/trec_setup.sh /media/jiho/datasets/TREC/2015/pool'

  This will read files and create collection.spec file in etc directory. Only
the files listed in the collection.spec file will be indexed. Then terrier
settings must be amended accordingly, edit etc/terrier.properties

For TREC documents, I use simpleXMLCollection format. Append below code and fix
them as necessary.

```
###################################################################
#SimpleXMLCollection specific properties
###################################################################
trec.collection.class=SimpleXMLCollection

#what tag defines the document
xml.doctag=DOC
#what tag defines the document number
xml.idtag=DOCNO
#what tags hold text to be indexed
xml.terms=KEYWORDS,BODY,REF,TITLE
#will the text be in non-English?
#string.use_utf=false
```

2. run indexer
    'bin/trec_terrier.sh -i'
    Before you run indexer, pre-existing index results and data must be removed
which are stored in var/

3. verify
    'bin/trec_terrier.sh --printstats'

Querying
========

run 'bin/interactive_terrier.sh' to query the index for results
