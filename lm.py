"""
LM.PY
Robin Melnick

Train a language model then calculate and report perplexity of 
the model as applied to separate test data.

NOTE: Initial supplied code trains a basic bigram model, NO smoothing, 
NO start and end-sentence tokens (e.g., <s>, </s>), NO support 
for unknown (previously unseen) words (unigrams) or bigrams.

See "TODO" sections for areas of code to update to implement
more sophisticated language models.
"""

from __future__ import division
import io
import math
import sys

"""
TODO: The following dictionaries (hash tables) are global 
stores for the critical language model data--counts and 
probabilities. Use as-is or modify as required for your 
implementation.
"""

unigram_counts = {}
unigram_probs = {}
bigram_counts = {}
bigram_probs = {}

"""
TODO: The following several functions build the language
model from the training data. Modify as necessary.
"""

"""
COUNT: 
-- Takes a list of bigram tuples (e.g., [['a','b'],['b','c'],['c','d']...])
-- increments unigram counts for first word, adding keys to dictionary
   (hashtable) for words seen for the first time
-- buids a dictionary of dictionaries (hash of hashes), adding keys as necessary 
   and incrementing bigram counts for exisitng items.
"""
def count(tuples):
    for bigram in tuples:
        x = bigram[0]
        y = bigram[1]

        # increment unigram counts for first word of pair, add key if necessary
        if x in unigram_counts:
            unigram_counts[x] += 1
        else:
            unigram_counts[x] = 1
            
        # increment bigram count, initialize one or both levels of keys if necessary
        if not x in bigram_counts:
            bigram_counts[x] = {} 
        if y in bigram_counts[x]:
            bigram_counts[x][y] += 1
        else:
            bigram_counts[x][y] = 1 

"""
CALC_UNIGRAM_PROBS: Call after unigram counting completed. 
"""
def calc_unigram_probs():
    n = sum(unigram_counts.values())
    for unigram in unigram_counts:
        unigram_probs[unigram] = unigram_counts[unigram] / n
            
"""
CALC_BIGRAM_PROBS: Call after bigram counting completed.
"""
def calc_bigram_probs():
    for x in bigram_counts:
        if not x in bigram_probs:
            bigram_probs[x] = {}
        for y in bigram_counts[x]:
            bigram_probs[x][y] = bigram_counts[x][y] / unigram_counts[x]
            
"""
TRAIN: Reads in supplied training data file, splits lines on white space,
and builds unigram and bigram language models using functions above.
            
As-is, supplied code does not implement start and end-of-sentence tokens 
(e.g., <s>, </s>) or <UNK> token.
"""
def train(training_file):
    f_train = io.open(training_file, 'r')
    
    # build up counts
    for line in f_train:
        words = line.strip().split()
        count(zip(words, words[1:]))
    
    # counts completed, now calculate the probabilities
    calc_unigram_probs()
    calc_bigram_probs()

"""
GET_BIGRAM_PROB: Takes a bigram (tuple) as input and retrieves the
calculated probability from the language model.

CRITICAL TODO NOTES:
    -- Function PERPLEXITY relies on GET_BIGRAM_PROB to retrieve calculated 
       probabilities during testing phase. If you modify the structure of 
       the language model (the bigram dictionary of dictionariess)
       be sure to update this funciton appropriately!
    -- As-is, returns zero for previously unseen bigrams--no smoothing,
       backoff, or interpolation--which, as-is, will in turn trigger an error-exit
       in PERPLEXITY if anything's encountered in testing that wasn't seen
       in training (since the log of zero is undefined).
"""
def get_bigram_prob(bigram):
    x = bigram[0]
    y = bigram[1]
    if (x in bigram_probs) and (y in bigram_probs[x]):
        return(bigram_probs[x][y])
    else:
        return(0)


"""
PERPLEXITY: Reads in testing file, retrieves bigram probabilities from the
language model using above function RETURN_BIGRAM_PROB (which you will need
to modify) and calculates perplexity for the language model as applied to
the supplied testing data
"""
def perplexity(testing_file):
    f_test = io.open(testing_file, 'r')
    n = 1
    log_prob_sum = 0
    for line in f_test:
        unigrams = line.strip().split()
        for bigram in zip(unigrams, unigrams[1:]):
            n += 1
            bigram_prob = get_bigram_prob(bigram)
            if bigram_prob == 0:
                sys.exit("Error: Bigram probability equals zero, terminating!")
            else:
                log_prob_sum += math.log10(bigram_prob)
    return abs(log_prob_sum / n)


"""
You should not need to edit this function.

Takes in the training and testing files and calls functions above to
train a language model then compute and output a perplexity score.
"""
def main(training_file, testing_file):
    train(training_file)
    print ("Perplexity: %.2f" % perplexity(testing_file))


"""
You should not need to edit this function.

Commandline interface takes the names of a training file and test file.
"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('Usage:\tlm.py <training file> <test file>')
        sys.exit(0)
    main(sys.argv[1],sys.argv[2])
