# Noise-Aware Alignment
Aligning Vector-spaces with Noisy Supervised Lexicons

This is an open source implementation of our framework using the noise aware aligment method:

Noa Yehezkel Lubin, Jacob Goldberger, and Yoav Goldberg. 2019. 
Aligning Vector-spaces with Noisy Supervised Lexicons. 
In Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics.

# Using Noise-Aware Alignment
The mapping algorithm is easy to use. All you need is noise_aware.py file.
Example to align X & Y matrices:

```
from noise_aware import noise_aware
transform_matrix, alpha, clean_indices, noisy_indices = noise_aware(X,Y)
```
* transform_matrix - The matrix solution of the aligment problem using noise-aware aligment.
* alpha - the percentage of clean indices.
* clean_indices - list of clean indicesvused in the aligment.
* noisy_indices - list of noisy indices ignored in the aligment.


# Reproducing Experiments
To recreate the two real world experiments described in the paper.

## Bilingual Word Embedding
this experiment is based on: Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2018.
[A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings](https://aclweb.org/anthology/P18-1073).

1. download [data](https://github.com/artetxem/vecmap/blob/master/get_data.sh) 
2. run:
```
python map_embeddings_naa.py --unsupervised <path/to/input_src> <path/to/input_trg> <path/to/output_src> <path/to/output_trg> -hem
```
3. evaluation is done with [vecmap](https://github.com/artetxem/vecmap) 

## Diachronic (Historical) Word Embedding
William  L  Hamilton,  Jure  Leskovec,  and  Dan  Juraf-sky. 2016.  
[Diachronic word embeddings reveal sta-tistical  laws  of  semantic  change.](http://www.aclweb.org/anthology/P16-1141)

1. donwload [english fiction data](http://snap.stanford.edu/historical_embeddings/eng-fiction-all.zip) and [frequencies](http://snap.stanford.edu/historical_embeddings/eng-all/freqs.pkl)
2. set path to downloaded historical embeddings in the jupyer notebook.
3. run notebook: Diachronic (Historical) Word Embeddings.ipynb

# Publications
If you use this software for academic research, please cite the relevant paper:
```
@inproceedings{yehezkel-lubin-etal-2019-aligning,
    title = "Aligning Vector-spaces with Noisy Supervised Lexicon",
    author = "Yehezkel Lubin, Noa  and
      Goldberger, Jacob  and
      Goldberg, Yoav",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1045",
    pages = "460--465",
    abstract = "The problem of learning to translate between two vector spaces given a set of aligned points arises in several application areas of NLP. Current solutions assume that the lexicon which defines the alignment pairs is noise-free. We consider the case where the set of aligned points is allowed to contain an amount of noise, in the form of incorrect lexicon pairs and show that this arises in practice by analyzing the edited dictionaries after the cleaning process. We demonstrate that such noise substantially degrades the accuracy of the learned translation when using current methods. We propose a model that accounts for noisy pairs. This is achieved by introducing a generative model with a compatible iterative EM algorithm. The algorithm jointly learns the noise level in the lexicon, finds the set of noisy pairs, and learns the mapping between the spaces. We demonstrate the effectiveness of our proposed algorithm on two alignment problems: bilingual word embedding translation, and mapping between diachronic embedding spaces for recovering the semantic shifts of words across time periods.",
}
```
