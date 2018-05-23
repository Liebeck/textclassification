# textclassification
This is a partially refactored version of the [ArgMining Repository](https://github.com/Liebeck/ArgMining) and the code for the argument mining tasks that were researched by Matthias Liebeck as part of his PhD thesis.

## Requirements
* [docker](https://www.docker.com/)


## Argument Mining Experiments
This repository contains code for the argument mining tasks machine learning experiments in the field of argument mining that

* Execute the *data/download_resources.sh* script to download the models
``` bash
./data/download_resources.sh data
```
* Depending on whether you want to use our trained Wikipedia word2vec models, you also need to download word embeddings via this script:
``` bash
./data/download_embeddings.sh data
```

* If you want to use character embeddings from fasttext, you can download the specific model here: http://lager.cs.uni-duesseldorf.de/NLP/fasttext/german/wikipedia/de-wiki_20170501/  
With the following script, you can download character (3, 3)-grams that were trained with 5 and with 50 iterations:

``` bash
./data/download_character_embeddings.sh data
```