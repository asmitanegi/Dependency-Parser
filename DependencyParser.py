import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util


"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon
"""


wordDict = {}
posDict = {}
labelDict = {}
system = None

def genDictionaries(sents, trees):
    """
    Generate Dictionaries for word, pos, and arc_label
    Since we will use same embedding array for all three groups,
    each element will have unique ID
    """
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n+1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]

def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]

def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]

def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """

def genTrainExamples(sents, trees):
    """
    Generate train examples
    Each configuration of dependency parsing will give us one training instance
    Each instance will contains:
        WordID, PosID, LabelID as described in the paper(Total 48 IDs)
        Label for each arc label:
            correct ones as 1,
            appliable ones as 0,
            non-appliable ones as -1
    """
    numTrans = system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = system.initialConfiguration(sents[i])

            while not system.isTerminal(c):
                oracle = system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = system.transitions[j]
                    if t == oracle: label.append(1.)
                    elif system.canApply(c, t): label.append(0.)
                    else: label.append(-1.)

                features.append(feat)
                labels.append(label)
                c = system.apply(c, oracle)
    return features, labels

def forward_pass(_, _, _):

    """
    =======================================================

    Implement the forwrad pass described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =======================================================
    """

if __name__ == '__main__':

    # Load all dataset
    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    # Load pre-trained word embeddings
    dictionary, word_embeds = pickle.load(open('word2vec.model', 'rb'))

    # Create embedding array for word + pos + arc_label
    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary: index = dictionary[w]
            elif w.lower() in dictionary: index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size)*0.02-0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    # Get a new instance of ParsingSystem with arc_labels
    system = ParsingSystem(labelDict.keys())

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."

    graph = tf.Graph()

    with graph.as_default():

        embeddings = tf.Variable(embedding_array, dtype=tf.float32)

        """
        ===================================================================

        Define the computational graph with necessary variables.
        You may need placeholders of:
            train_inputs
            train_labels
            test_inputs

        Implement the loss function described in the paper

        ===================================================================
        """

        optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)

        # Compute Gradients
        grads = optimizer.compute_gradients(loss)
        # Gradient Clipping
        clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
        app = optimizer.apply_gradients(clipped_grads)

        init = tf.global_variables_initializer()

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:
        init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step*Config.batch_size)%len(trainFeats)
            end = ((step+1)*Config.batch_size)%len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, loss_val = sess.run([app, loss], feed_dict=feed_dict)
            average_loss += loss_val

            # Display average loss
            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0

            # Print out the performance on dev set
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = system.numTransitions()

                    c = system.initialConfiguration(sent)
                    while not system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(test_pred, feed_dict={test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and system.canApply(c, system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = system.transitions[j]

                        c = system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Optimization Finished."

        print "Start predicting on test set"
        predTrees = []
        for sent in testSents:
            numTrans = system.numTransitions()

            c = system.initialConfiguration(sent)
            while not system.isTerminal(c):
                feat = getFeatures(c)
                pred = sess.run(test_pred, feed_dict={test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and system.canApply(c, system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = system.transitions[j]

                c = system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Store the test results."
        Util.writeConll('result.conll', testSents, predTrees)

