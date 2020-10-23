This directory produces a JSON document with interpretable results of [the Soft
Cosine Measure (SCM) math information retrieval system at ARQMath 2020][paper].

 [paper]: http://ceur-ws.org/Vol-2696/paper_235.pdf#page=10

## Installation

To install the required packages, use Python 3 and execute the following
commands in this directory:

``` sh
git submodule update --init --recursive
pip install -r SCM-at-ARQMath/input_data/requirements.txt
pip install -r SCM-at-ARQMath/requirements.txt
```

## Downloading the input data

To download the required input data, execute the following commands in this directory:

``` sh
dvc pull
```

## Producing the JSON document

TODO
