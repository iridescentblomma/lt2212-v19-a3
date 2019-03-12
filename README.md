# LT2212 V19 Assignment 3

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: Iris Epple

## Additional instructions

Give outputfile argument in gendata without file ending.

In train and test give outputfile for model without file ending.

For POS option provide the flag -P for gendata.py

## Reporting for Part 4

Hypothesis:

Accuracy increases and perplexity decreases as with of ngrams and/or number of lines increase.

Outputs:

| commandline                | Accuracy | Perplexity |
|----------------------------|----------|------------|
| -S 500 -E 1000 -T 100      | 0.125    | 66.4       |
| -N 5 -S 500 -E 1000 -T 100 | 0.126    | 63.99      |
| -S 200 -E 1000 -T 160      | 0.13     | 69.04      |
| -N 5 -S 200 -E 1000 -T 160 | 0.13     | 68.18      |
| -E 500 -T 100              | 0.12     | 67.83      |
| -N 2 -E 500 -T 100         | 0.12     | 68.88      |
| -N 5 -E 500 -T 100         | 0.12     | 69.84      |
| -S 1000 -E 1100 -T 20      | 0.11     | 46.08      |
| -N 5 -S 1000 -E 1100 -T 20 | 0.19     | 45.75      |


Interpretation:

The outcomes do most of the time not support the hypothesis. Whereas the accuracy goes up 
and the perplexity goes down as the width of ngrams increases, better results are observed with
less data rather than more data, which is surprising.


## Reporting for Part Bonus 


| commandline                  | Accuracy | Perplexity |
|------------------------------|----------|------------|
| -S 500 -E 1000 -T 100 -P     | 0.277    | 8.27       |
| -N 5 -S 500 -E 1000 -T 100 -P| 0.266    | 8.17       |

| -S 200 -E 1000 -T 160 -P     | 0.279    | 8.21       |
| -N 5 -S 200 -E 1000 -T 160 -P| 0.273    | 7.85       |

| -E 500 -T 100 -P             | 0.274    | 8.56       |
| -N 2 -E 500 -T 100 -P        | 0.264    | 9.10       |
| -N 5 -E 500 -T 100 -P        | 0.257    | 8.257      |

| -S 1000 -E 1100 -T 20 -P     | 0.231    | 9.22       |
| -N 5 -S 1000 -E 1100 -T 20 -P| 0.270    | 9.29       |


Compared to the words without POS the accuracy is very high and perplexity very low.
Which means that the encoding of the words alongside with their POS and the POS as class labels
is much better. This is not too surprising since the encoding of the words alongside their POS
is much more precise than "just" the word. Furthermore, using POS tags as class labels 
increases the probability of "getting it right", since there are way less POS tags than 
different words in the corpus.
But here, in contrast to the words, the ngram width has different impact on the results
given different sizes of data. With 500 lines, accuracy decreases with higher ngram width, it 
stays approximately the same for 800 lines and it actually increases with only 100 lines.
This could be the case that with little data the model relies more on the context to make
predictions, whereas with a lot more date, a greater ngram width would impose to many 
restrictions, which means, the likelihood of getting exactly that sequence is not very high.

