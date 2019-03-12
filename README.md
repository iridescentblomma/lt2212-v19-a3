# LT2212 V19 Assignment 3

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: Iris Epple

## Additional instructions

Give outputfile argument in gendata without file ending.

In train and test give outputfile for model without file ending.

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

(Delete if you aren't doing the bonus.)
