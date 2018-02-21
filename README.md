LGCS 129
Robin Melnick
Homework 2b
Language Modeling—Implementation
DUE: Thursday, 2/22/18
For the main part of this assignment you will be constructing smoothed versions of a bigram language model and evaluating performance of these models intrinsically using perplexity. I have provided some Python starter code and have also given you a specification to follow that will help guide you through the process and also make grading easier. You will be submitting your code and a short write-up.
You may collaborate with classmates in your work — in fact, it’s encouraged! — but just as with writing prose, you each need to ultimately complete and submit your own work, not simply identical copies of a single file.
1. Getting Started
First, read through this entire handout before getting started!
Next, you can find the starter code and training data for this project in either of two places—as
a zipped archive on Sakai or on our class server in: /data/classes/129/homework/hw2
The starter material includes a file called sentences that contains 110,000 English sentences that you will be using to train and evaluate your language model. These sentences were taken from Simple English. Most of the preprocessing work has been done for you, so you can identify individual tokens/words by just splitting on whitespace. You’ll also notice that the text has been normalized in some cases, for example by lowercasing and converting numbers to a special token.
2. Code
All of your code for this assignment should be contained with a pair of modified versions of the file lm.py. See especially the areas marked as TODO in the embedded comments for notes on which sections you should change and which areas you should leave untouched.
As-is, the existing lm.py starter code takes two command-line arguments, a training data file and a testing data file, reads in the training data, and then builds a simple MLE bigram language model (no start and end-of-sentence markers, no unknown-word support, no smoothing, backoff, or interpolation). The code then calculates and reports the perplexity measure (related to average bigram probability) for the model as applied to the testing data provided.
1 of 6
3. Task #1 — Add-k Smoothing
Update (or replace) the bigram language model to deal with as-yet unseen items by implementing smoothing, adding a small value k to the frequency counts of each bigram. (Recall that when k=1, this is known as Laplace or Add-1 smoothing.)
Requirements:
• Modify the starter code any way you like, so long as you don’t change main() — i.e., your program should still take the same two expected command-line arguments (a training filename and a testing filename) and should still print just the single output perplexity result line you’ll find in the existing main(). You’re welcome to add other printing for debugging and/or testing, but delete or comment-out such lines before your final submission.
• Use the symbol <s> to mark the beginning of sentences and </s> to mark the end. You’ll need to add these within your code since the source data will not include them. Don’t forget to also do this in the testing portion (for example, possibly in the perplexity function) for when you’re given new sentences in the testing phase.
• Build a fixed vocabulary based on the training data provided and use the <UNK> symbol to mark unknown (out-of-vocabulary) words. Specifically:
During training (i.e., when you are learning the probabilities)
o Youshouldreplacethefirstoccurrenceofeachwordwith<UNK>andtrainthe
model using that.
o <s> and </s> will be part of your vocabulary, but do NOT replace the first
occurrences of <s> and </s>.
For example, if you were given three sentences (I’ve used letters for words):
aaab abba caaa
this would become
<s> <UNK> a a <UNK> </s> <s> a b b a </s> <s><UNK> a a a </s>
and then you’d use this data to train your model. Here, this would result in a vocabulary of 5 words, specifically <s>, <UNK>, a, b, and </s>.
2 of 6
During testing (i.e., in the perplexity and return_bigram_prob functions)
• If you now see a word that was not seen during training, you should replace it with the <UNK> symbol during testing. Notice that if you had previously seen a particular word just once during training, it would not end up present in your trained vocabulary, and if appearing now in testing, would now be treated as an unknown word. See pg. 95 from the textbook for a description of this approach.
For example, after training on the imaginary three-sentence corpus just discussed above, in testing the sentence
abaacad
would become
<s> a b a a <UNK> a <UNK> </s>
• As noted above, smooth bigrams using an add-k factor. You can start out using k=1 (Laplace smoothing), but note that later in the assignment, we’ll discuss playing with different values for k. Also, note that because we’re using the <UNK> symbol, you should NOT smooth the unigrams.
Hints/Advice
• Before testing with larger data sets, start by working just with, for example, the three- sentence corpus from homework 2a. During development, you can add extra print lines—or use the tracing capabilities of your favorite Python IDE—to check your probabilities against hand-worked examples. You might also want to do a few extra examples by hand to test other cases.
• To fully test smoothing, of course, you need to encounter items in your test data that were not present in your training data. Try working with random samples pulled from the large sentences file. To get a random subset of, say, 1,000 lines from the larger data set, on the unix/linux/terminal command line, try the following:
sort -R sentences | head -n 1000
or replace “1000” here with however many lines you want. Pull one sample to train with and another to test against. (Later you’ll work with the full set—see below.)
(Do note that sorting the full 110,000-line file takes about 20 seconds on my laptop, could be more or less on yours—so don’t worry that you don’t get an instantaneous response.)
• There are many ways to structure your training code, but as one suggestion, the
3 of 6
provided starter code chooses to first run through the data, recording all of the counts, before going back through the stored counts to calculate the probabilities.
• You are free to specify your data structures as you like. You could, for example, use a single dictionary with the key being the bigram. The starter code uses dictionaries (hash tables), and in particular implements the bigram counts as a dictionary of dictionaries, where the main dictionary is keyed off of the first word and the value is another dictionary. The second dictionary is keyed off of the second word in the bigram and has the count (or probability) as the value. This approach is likely to make later parts of this assignment—absolute discounting—much easier.
• Be careful about underflow. Notice that the current perplexity-calculation code does not take the log of a product of many probabilities, but rather the (mathematically equivalent) sum of the logs of the individual probabilities.
• On my laptop, training and testing my implementation of add-k smoothing on the full 110,000-line sentences file finishes in about 15 seconds. If you find that yours is taking a long time (i.e., much more than a minute or so) you’re probably doing something inefficiently.
4. Task #2 — Absolute Discounting
The language model above doesn’t use the unigram probabilities at all. Let’s try to improve this with a language model that backs off to the unigram probabilities. Specifically, you should construct an absolute discounted backoff language model. We did this by hand in the last part of hw2a. (Also see the posted lecture slides, the handout with worked example, and pg. 110 in the textbook.)
Your implementation should include the following:
• Initially try setting your discount rate to D = 0.75, subtracting this amount from the bigram probability for each bigram type seen in training. (Later below we’ll play with varying D to gauge its effect on model quality, as evaluated by perplexity.)
• Like the add-k language model above, enclose sentences in <s> and </s> and use a closed vocabulary with the same approach for using <UNK> as in the add-k model.
• Unigram probabilities should be calculated normally. Note that <s>, </s> and <UNK> all have their own unigram probabilities since they are part of the vocabulary.
• As in our hand-worked examples, when calculating bigram probabilities during training, you discount each actual bigram count by subtracting D. During testing, if you encounter an unseen bigram, you calculate its probability as the backoff factor α times the unigram probability of the second word in the bigram. α is different for each word
4 of 6
and will depend on how many bigrams were previously discounted, i.e., those that started with the first word in the unseen bigram. (Again, see our prior hand-worked examples.)
It turns out that representing the bigram probabilities as a dictionary of dictionaries (hash of hashes), as I’ve suggested above, makes identifying the required data for these calculations easier.
Hints/Advice
• As before, I strongly encourage you to work out the probabilities by hand on a small example and then compare them to the system’s output. Think simple, with just a few sentences of length 4-5 and a vocabulary of just 3-4 words.
• If your training for the first language model was done in two steps, first collecting the counts, then going back and calculating the probabilities, I would encourage you to reuse code.
• Training and testing this language model on 100K+ sentences also runs in about 20 seconds for my implementation on my laptop. Again, if yours takes significantly longer than this (i.e., minutes) then you’re likely doing something inefficiently.
5. Evaluation
Now that we have a couple of working systems, let’s evaluate how well each does while varying a few parameters. Include a brief write-up describing the results from the experiments below.
Begin by splitting the sentences into two parts: 100,000 lines for training, 10,000 for testing. The easiest way to do this is probably on the unix/linux/terminal command line using the head and tail commands:
head -n 100000 sentences > training_filename tail -n 10000 sentences > testing_filename
QUESTIONS
a. What is the best value for k, the smoothing parameter in Task 1? Calculate the perplexity on the test set for k in (0.1, 0.01, 0.001, ..., 0.00000001) and provide the results in your write-up (either as a table or a graph).
b. What is the best discount parameter D in Task 2? Calculate perplexity on the test set for D in (0.99, 0.9, 0.75, 0.5, 0.25, 0.1) and provide the results and analysis in your write-up.
5 of 6
c. Which model is better? Try each with a few other-sized training/testing splits. What conclusions do you draw?
d. Very briefly answer the following questions: How long did you spend on this assignment? How would you improve it if I had to give it again?
6. Submitting
Create a zip archive, with name hw2b.zip, and with the following three files, with names as
specified here:
a. lm.addk.py — source code for Task 1, add-k smoothing language model
b. lm.abs.py — source code for Task 2, absolute discounting language model
c. hw2.writeup (.txt, .doc/x, or .pdf) — your write-up per Section 5 above
Post your zip archive to your Dropbox on the class Sakai site. 7. Commenting and code style
Your code should be commented appropriately. (Though you don’t need to go overboard!) The most important things:
• Your name and the assignment number should be at the top of each file.
• Each function or method should have an appropriate header/introductory comment.
• If anything is complicated, put a short note in there to help me out!
This is a non-trivial assignment and it can get complicated, which makes code clarity and comments very important so that I can understand what you did. For this reason, you will lose points for poorly commented or poorly organized code.
8. Grading
25% each for:
i. add-k model
ii. absolute discounting model
iii. evaluation/write-up iv. style/commenting
