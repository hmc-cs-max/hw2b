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
#this set is to control the unknown operator. I do not want markers of punctuation to be replaced with unknowns
train_words = {'<s>', '</s>', '.'}
#this holds the vocabulary
vocab = {'<s>', '</s>'}


#This is the place where K lives, and can be changed
k = 1
"""
TODO: The following several functions build the language
model from the training data. Modify as necessary.
"""


"""
CLEAN_TRAIN_LINE: 
--takes in a list, that is a single line in the file
--begins the line with the sentence marker
-- uses the seen_words set to determine if a word has been seen before, and should be replaces with <UNK>
-- removes period not attached to word, replaces with sentence markers
"""
def clean_train_line(test_line):
    test_line.insert(0,'<s>')
    i = 0
    while i < len(test_line):
        word = test_line[i]

        #replace period with sentence markers, if not end of line add new sentence marker
        if word == ".":
            test_line[i] = '</s>'
            if i+1 < len(test_line):
                test_line.insert(i+1, '<s>')

        #if word has not been seen before, add it to theseen list and replace with <UNK>
        if not word in train_words:
            train_words.add(word)
            test_line[i] = '<UNK>'
        else:
            vocab.add(word)


        #increment i
        i += 1

""""
CLEAN_TEST_LINE: 
--takes in a list, that is a single line in the file
--begins the line with the sentence marker
-- uses the seen_words set to determine if a word has been seen before, and should be replaces with <UNK>
-- removes period not attached to word, replaces with sentence markers
"""
def clean_test_line(test_line):
    test_line.insert(0,'<s>')
    i = 0
    while i < len(test_line):
        word = test_line[i]

        #replace period with sentence markers, if not end of line add new sentence marker
        if word == ".":
            test_line[i] = '</s>'
            if i+1 < len(test_line):
                test_line.insert(i+1, '<s>')

        #if word has not been seen before, add it to theseen list and replace with <UNK>
        if not word in vocab:

            test_line[i] = '<UNK>'
        


        #increment i
        i += 1




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
--smoothing is implemented here for those which require it
"""
def calc_bigram_probs():
    for x in bigram_counts:
        if not x in bigram_probs:
            bigram_probs[x] = {}
        for y in bigram_counts[x]:
            print(x,y)
            bigram_probs[x][y] = (bigram_counts[x][y] +k) / (unigram_counts[x] + k*len(vocab))
            
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
        # print(words)
        clean_train_line(words)
        # print(words) 

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
        print("from bigram table", bigram_probs[x][y])
        return(bigram_probs[x][y])

    elif (x in bigram_probs):
        laplaceBigram = 1/(unigram_counts[x]+len(vocab))
        print ('from get_probs', laplaceBigram)
        return laplaceBigram

    else:
        laplaceBigram = 1/(len(vocab))
        print ('from get_probs', laplaceBigram)
        return laplaceBigram



"""
PERPLEXITY: Reads in testing file, retrieves bigram probabilities from the
language model using above function RETURN_BIGRAM_PROB (which you will need
to modify) and calculates perplexity for the language model as applied to
the supplied testing data
"""
def perplexity(testing_file):
    print('vocab = ', vocab)
    print('unigram counts:, ', unigram_counts)
    print('\n')
    print('unigram probs: ', unigram_probs)
    print('\n')
    print('bigram counts: ', bigram_counts)
    print('\n')
    print('bigram probs: ', bigram_probs)

    f_test = io.open(testing_file, 'r')
    n = 1
    log_prob_sum = 0
    for line in f_test:
        unigrams = line.strip().split()
        #check if the words are in the vocab
        clean_test_line(unigrams)
        for bigram in zip(unigrams, unigrams[1:]):

            print(bigram)
            n += 1
            bigram_prob = get_bigram_prob(bigram)
            print('prob', bigram_prob)
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
