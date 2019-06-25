# Cross-domain Authorship Attribution: Author Identification using a Multi-Aspect Ensemble Approach
This project contains our paper's codes in python, used to indetifying authors in [PAN 2019](https://pan.webis.de/clef19/pan19-web/author-identification.html) competition.
# Usage

Parameters:
```
authorship_attribution.py [-h] [-i INPUT] [-o OUTPUT] [-n N] [-ft FT]
                                 [-pt PT]
optional arguments:
  -h, --help                      show this help message and exit
  -i INPUT, --input INPUT         path to input dataset
  -o OUTPUT, --output OUTPUT      path to output directory
  -n N                            n gram
  -ft FT                          frequency term for tfidf and ngram
  -pt PT                          threshold for UNK authors
```
and the default parameters are:
* n : `4`
* ft: `5`
* pt: `0.08`

running the script usage:
```
python authorship_attribution.py -i path_to_input_dataset_dir -o path_to_output_dir
```
# Citation
Please cite us as:

*M Rahgouy, HB Giglou, T Rahgouy, MK Sheykhlan, E Mohammadzadeh. Cross-domain Authorship Attribution: Author Identification using a Multi-Aspect Ensemble Approach - Notebook for PAN at CLEF 2019. In CLEF 2019 Evaluation Labs and Workshop–Working Notes Papers. CEUR-WS. org.*
