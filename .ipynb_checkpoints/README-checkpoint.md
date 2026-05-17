# Overview

In this assignment, you will gain hands-on experience with the implementation of a widely used machine learning algorithm - Naive Bayes - and apply it to a real-world classification task.

You are expected to implement the standard Naive Bayes algorithm, and possibly, explore enhancements to improve its performance. Once your model is trained and validated, you will then apply it on a test set with no class label and submit the predictions of your model to Kaggle. The system will evaluate your results with the ground-truth data and report the accuracy of your implementation.

This exercise is designed to help you:

    Understand the Naive Bayes algorithm through direct implementation.
    Experience working with real-world text data.
    Tackle practical challenges in machine learning, such as preprocessing and generalisation.

# The challenge

The goal of this programming assignment is to build a Naive Bayes classifier capable of predicting the category of points of interest - one of "Restaurants", "Shopping" and "Nightlife" - from the Yelp dataset. The Yelp dataset contains various attributes for each point-of-interest including the name, the GPS location (latitude and longitude), a piece of review, average check-in time (i.e. hour of the day). Your target is to predict the category of a given point-of-interest and achieve high predictive accuracy on the test data. In this assignment, you will need to complete the following tasks:

    Build a baseline model by implementing the Naive Bayes classifier from scratch, which is a standard version of Naive Bayes as explained in Mitchell's "Machine Learning" textbook, that runs only on the review attribute. For this task, do not use ready implementations of Naive Bayes such as the one available in the scikit-learn package. Represent each review by a word frequency vector (you may use a package for this step, e.g., CountVectorizer in scikit-learn). Record your prediction performance on the train/validation set as well as on the test set as shown on the public Kaggle leaderboard.  We have built such baseline and it listed as benchmark model on the Kaggle leaderboard. 
    Improve on the baseline model based on the review attribute only. You can consider improving the baseline using better data representation and preprocessing techniques. You can also consider extending the Naive Bayes classifier. Your implementation should still be Naive Bayes, which assumes that all attributes are conditionally independent given the class. The paper Rennie at al. (2003) "Tackling the Poor Assumptions of Naive Bayes Text Classifiers" lists several extensions for your reference. (You may use packages like scikit-learn for this step.)
    Improve your model by adding additional attributes to your model. You may need to first consider which attributes might be useful for the prediction and combine the additional attributes within the same framework of Naive Bayes, i.e., the attributes are independent of each other, and they are independent of the words in the review given a particular class. By incorporating additional attributes, the prediction accuracy should be better than the model you developed in task 2. Note, some of the attributes are real-valued, you will need to think how to handle them if you include them: this is explorative part of the project, there can be multiple ways to do implement it; make sure you explain your proposed solution.

When you build your Naive Bayes classifiers, you might consider using cross-validation to pick the best values for the hyperparameters of your preprocessing technique and model. You can use numpy, pandas, nltk and sklearn for data prepossessing/representation, model evaluation/validation in all tasks. 
## Tips

    Compute probabilities in log space and add them instead of multiplying them; log(ab) = log(a) + log(b). The result will be the same in theory but multiplying tiny numbers can give computational issues.
    Use Laplace smoothing to avoid zero counts.  However, neither Laplace smoothing nor log-calculations count as an extension.
