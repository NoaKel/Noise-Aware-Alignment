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
python map_embeddings_latent.py <path/to/src> <path/to/trg> <src_language_name> <trg_language_name> -hem
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

