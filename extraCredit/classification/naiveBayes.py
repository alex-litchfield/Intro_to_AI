# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        "*** YOUR CODE HERE ***"

        # featureLabels: ((label, feature), count)
        # featureValues: ((label, feature, value), count)
        featureLabels, featureValues, optimalConditionalProbabilities = dict(), dict(), dict()
        labelCounts, self.prior = util.Counter(), util.Counter()
        optimalK, optimalAccuracy = None, -1
        # Creating initial counts/frequencies
        for i in range(len(trainingData)):
            currlabel = trainingLabels[i]
            for j in self.features:
                # If the value is not already there we initialize it in both dictionaries as 0
                for k in range(0, 2): # Accounts for binary values
                    if (currlabel, j, k) not in featureValues:
                        featureValues[(currlabel, j, k)] = 0
                if (currlabel, j) not in featureLabels:
                    featureLabels[(currlabel, j)] = 0
                # Incrementing each occurence in the dictionaries
                featureValues[(currlabel, j, trainingData[i][j])] = featureValues[(currlabel, j, trainingData[i][j])] + 1
                featureLabels[(currlabel, j)] = featureLabels[(currlabel, j)] + 1
            labelCounts[currlabel] = labelCounts[currlabel] + 1
        # Estimate and track conditional probabilities from training data for each possible value of k given in the kgrid list
        for kInGrid in kgrid:
            prior = util.Counter()
            conditionalProbabilities = dict()
            for i in self.legalLabels:
                for j in self.features:
                    for k in range(0, 2): # Accounts for binary values
                        # Apply Laplace smoothing when estimating the current prior
                        conditionalProbabilities[(i, j, k)] = (featureValues[(i, j, k)] + kInGrid) / (featureLabels[(i, j)] + 2 * kInGrid)
                # Actually estimating the prior
                prior[i] = labelCounts[i] / sum(labelCounts.values())
            # Classify the validation set
            self.prior, self.conditionalProbs = prior, conditionalProbabilities
            predictions = self.classify(validationData)
            correctPredictions = 0
            for i in range(len(validationLabels)):
                if predictions[i] == validationLabels[i]:
                    correctPredictions = correctPredictions + 1
            predictionAccuracy = correctPredictions / len(validationLabels)
            # Finding the optimalK value based on the predictionAccuracy above
            if predictionAccuracy > optimalAccuracy or ((kInGrid < optimalK or optimalK is None) and predictionAccuracy == optimalAccuracy):
                optimalConditionalProbabilities, optimalPrior, optimalAccuracy, optimalK = conditionalProbabilities, prior, predictionAccuracy, kInGrid
        # Storing information in the class to be accessed later
        self.conditionalProbs, self.prior, self.k = optimalConditionalProbabilities, optimalPrior, optimalK


    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"

        # Finding log distribution over all the legal labels
        for i in self.legalLabels:
            # Initialize the log probability
            logProbability = math.log(self.prior[i])
            # Apply the probability/datam of each feature to the log probability
            for j in self.features:
                currentProbability = self.conditionalProbs[(i, j, datum[j])]
                logProbability = logProbability + math.log(currentProbability)
            # Placing the logProbability into the log joint distribution
            logJoint[i] = logProbability
        # Return log joint distribution
        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []
        "*** YOUR CODE HERE ***"
        
        # Finds the odds ratio for each feature and puts them in featuresOdds
        for i in self.features:
            featuresOdds.append((self.conditionalProbs[(label1, i, 1)] / self.conditionalProbs[(label2, i, 1)], i))
        # Sorts featuresOdds in descending order (greatest value first)
        featuresOdds.sort(reverse=True)
        # Holds the top 100 features only
        featuresWithHighestOdds = []
        # Loops through the first 100 features (highest 100) and appends to featuresWithHighestOdds list
        for j in featuresOdds[:100]:
            featuresWithHighestOdds.append(j[1])
        # Returns the first 100 features (highest 100)
        return featuresWithHighestOdds
