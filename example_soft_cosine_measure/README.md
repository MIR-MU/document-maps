This directory contains a JSON document with interpretable search results of
[the Soft Cosine Measure (SCM) math information retrieval system at ARQMath
2020][paper].

 [paper]: http://ceur-ws.org/Vol-2696/paper_235.pdf#page=10

# Reproducing the JSON document

If you would like to reproduce the JSON document, follow the steps in this
section. *This will require a lot of time and storage space and it is not
recommended for most users.*

To install the required packages, use Python 3 and execute the following
commands in this directory:

``` sh
$ git submodule update --init --recursive
$ pip install -r scm_at_arqmath/input_data/requirements.txt
$ pip install -r scm_at_arqmath/requirements.txt
```

To download the required input data, execute the following command in this
directory. *Up to 1TiB of data will be downloaded.*

``` sh
$ dvc pull
```

To produce the JSON document, execute the following command in this directory.
Note that this will take up to 30 minutes.

``` sh
$ python -m scripts.produce_json_document > document.json
```

The JSON document `document.json` with interpretable search results should now
be produced. The document should have no more than 100 MiB in size.

# Interpreting the JSON document

The JSON document consists of the following sections: `results`, `texts`,
`dictionary`, `word_similarities`, and `texts_bow`, which I will describe
below together with Python code examples.

``` python
>>> import json
>>>
>>> with open('document.json', 'rt') as f:
...     document = json.load(f)
```

For each query, the `results` section lists the first five search results
of the SCM math information retrieval system. For example, the first five
results for the query `A.1` are the documents `2805089`, `499578`, `374465`,
`132776`, and `427697`:

``` python
>>> document['results'].keys()
dict_keys(['A.1', 'A.10', 'A.100', ..., 'A.99'])
>>> document['results']['A.1']
['2805089', '499578', '374465', '132776', '427697']
```

The `texts` section contains the texts of queries and documents as word
identifiers.  For example, the query `A.1` and the document `2805089` contain
the following word identifiers:

``` python
>>> document['texts']['A.1']
['5375', '519', '51', ..., '1264']
>>> document['texts']['2805089']
['355', '4903', '147', ..., '280']
```

To make the texts human-readable, the `dictionary` section describes the
mapping between word identifiers and words. For example, the query `A.1` and
the document `2805089` contain the following texts:

``` python
>>> def word_id_to_word(word_id):
...     return document['dictionary'][word_id]
...
>>> def get_text(text_id):
...     text = map(word_id_to_word, document['texts'][text_id])
...     return ' '.join(text)
...
>>> get_text('A.1')
'finding value of V!C such that the range of the rational function ...'
>>> get_text('2805089')
'here am using the given definition of topologically transitive ...'
```

The `word_similarities` section lists the similarity of word pairs. For
example, we can see that the phrases for "Wacław Sierpiński" and "Kazimierz
Kuratowski" are considered slightly similar, because both Sierpiński and
Kuratowski are Polish mathematicians who co-authored several influential
papers. However, Sierpiński did not enjoy spicy meals:

``` python
>>> reversed_dictionary = {v: k for k, v in document['dictionary'].items()}
>>>
>>> def word_to_word_id(word):
...     return reversed_dictionary[word]
...
>>> def word_similarity(word1, word2):
...     word1_id = word_to_word_id(word1)
...     word2_id = word_to_word_id(word2)
...     if word1_id == word2_id:
...        return 1.0
...     if int(word1_id) > int(word2_id):
...        word1_id, word2_id = word2_id, word1_id
...     if word1_id not in document['word_similarities']:
...        return 0.0
...     if word2_id not in document['word_similarities'][word1_id]:
...        return 0.0
...     return document['word_similarities'][word1_id][word2_id]
...
>>> word_similarity('waclaw sierpinski', 'waclaw sierpinski')
1.0
>>> word_similarity('waclaw sierpinski', 'kuratowsky')
0.10005377233028412
>>> word_similarity('waclaw sierpinski', 'spicy dishes')
0.0
```

The `texts_bow` section contains [the bag of words representations][bow] of
queries and documents. For example, the query `A.1` and the document `2805089`
are represented as follows:

 [bow]: https://en.wikipedia.org/wiki/Bag-of-words_model

``` python
>>> def get_bag_of_words(text_id):
...     return {
...         word_id_to_word(word_id): weight
...         for word_id, weight
...         in document['texts_bow'][text_id].items()
...     }
...
>>> get_bag_of_words('A.1')
{'O!DIVIDE': 0.00631864214875213, 'for': 0.0012751444044670375, 'of': 0.0016497711588802603, ...} 
>>> get_bag_of_words('2805089')
{'an': 0.005152167526307428, 'and': 0.0015047092876798649, 'both': 0.0068842247445634165, ...}
```

Using the `texts_bow` and `word_similarities` sections, we can compute [the
soft cosine similarity measure][scm] between queries and documents. For
example, the query `A.1` and the document `2805089` are [fairly similar
(0.29)][landis-koch]:

 [landis-koch]: https://en.wikipedia.org/wiki/Cohen%27s_kappa#Interpreting_magnitude
 [scm]: https://en.wikipedia.org/wiki/Cosine_similarity#Soft_cosine_measure

``` python
>>> from math import sqrt
>>>
>>> def word_id_similarity(word1_id, word2_id):
...     word1 = word_id_to_word(word1_id)
...     word2 = word_id_to_word(word2_id)
...     return word_similarity(word1, word2)
...
>>> def inner_product(text1_id, text2_id):
...     text1_bow = document['texts_bow'][text1_id]
...     text2_bow = document['texts_bow'][text2_id]
...     text_similarity = 0.0
...     for word1_id, word1_weight in text1_bow.items():
...         for word2_id, word2_weight in text2_bow.items():
...             word_similarity = word_id_similarity(word1_id, word2_id)
...             text_similarity += word1_weight * word_similarity * word2_weight
...     return text_similarity
...
>>> def soft_cosine_measure_norm(text1_id, text2_id):
...     norm = 1.0
...     norm *= inner_product(text1_id, text1_id) or 1.0
...     norm *= inner_product(text2_id, text2_id) or 1.0
...     norm = sqrt(norm)
...     return norm
...
>>> def soft_cosine_measure(text1_id, text2_id):
...     text_similarity = inner_product(text1_id, text2_id)
...     text_similarity /= soft_cosine_measure_norm(text1_id, text2_id)
...     return text_similarity
...
>>> soft_cosine_measure('A.1', 'A.1')
1.0
>>> soft_cosine_measure('2805089', '2805089')
1.0
>>> soft_cosine_measure('A.1', '2805089')
0.2932158719357693
```

By inspecting the largest elements in [the soft cosine similarity
measure][scm], we can see the most important word pairs. For example, the most
important exact word matches that caused the document `2805089` to be the first
result for the query `A.1` are "contain", "rational", and `O!INTERVAL(C-C)`,
and the most important soft word matches are ("rational", "rationals"), (`V!F`,
`V!X`), and (`V!C`, `V!A`):


``` python
>>> def interpret_soft_cosine_measure(text1_id, text2_id):
...     text1_bow = document['texts_bow'][text1_id]
...     text2_bow = document['texts_bow'][text2_id]
...     word_pair_importances = dict()
...     for word1_id, word1_weight in text1_bow.items():
...         for word2_id, word2_weight in text2_bow.items():
...             word_similarity = word_id_similarity(word1_id, word2_id)
...             word_pair_importance = word1_weight * word_similarity * word2_weight
...             if word_pair_importance == 0:
...                 continue
...             word1 = word_id_to_word(word1_id)
...             word2 = word_id_to_word(word2_id)
...             if (word1, word2) not in word_pair_importances:
...                 word_pair_importances[word1, word2] = 0.0
...             word_pair_importances[word1, word2] += word_pair_importance
...     norm = soft_cosine_measure_norm(text1_id, text1_id)
...     normalized_word_pair_importances = {
...         (word1, word2): word_pair_importance / norm
...         for (word1, word2), word_pair_importance
...         in word_pair_importances.items()
...     }
...     return normalized_word_pair_importances
...
>>> def get_most_important_exact_matches(word_pair_importances):
...     word_pair_importances = {
...         word1: word_pair_importance
...         for (word1, word2), word_pair_importance
...         in word_pair_importances.items()
...         if word1 == word2
...     }
...     return sorted(word_pair_importances.items(), key=lambda x: x[1], reverse=True)
...
>>> def get_most_important_soft_matches(word_pair_importances):
...     word_pair_importances = {
...         (word1, word2): word_pair_importance
...         for (word1, word2), word_pair_importance
...         in word_pair_importances.items()
...         if word1 != word2
...     }
...     return sorted(word_pair_importances.items(), key=lambda x: x[1], reverse=True)
...
>>> word_pair_importances = interpret_soft_cosine_measure('A.1', '2805089')
>>> get_most_important_exact_matches(word_pair_importances)
[
    ('contain', 0.04991217404756538),
    ('rational', 0.04505503729652947),
    ('O!INTERVAL(C-C)', 0.02870048143104925),
    ...
    ('the', 0.00012159285330011778),
]
>>> get_most_important_soft_matches(word_pair_importances)
[
    (('rational', 'rationals'), 0.039287104391766275),
    (('V!F', 'V!X'), 0.0035892678150422734),
    (('V!C', 'V!A'), 0.00306887141995381),
    ...
    (('this', 'an'), 4.983855583091146e-06)
]
```

These most important exact and soft word matches are what we'd like to visualize.
