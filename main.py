import csv
import math
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer

# Filenames
all_training = "./dataset/train.csv"
all_test = "./dataset/test.csv"


# Data preprocessing
def extract(filename):
    """
    Extracts reviews as a vectorized np array and categories
    depending on the data 
    """
    reviews = []
    all_categories = []

    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader) # Skip the headers
        for row in csvreader:
            reviews.append(row[5])
            training = len(row) == 7 
            if training: 
                all_categories.append(row[-1])

    vectorizer = CountVectorizer()
    reviews_array = vectorizer.fit_transform(reviews).toarray()
    return {
        'categories': all_categories,
        'reviews': reviews_array
    }


# Calculate probabilities
def get_review_conds(categories, rv_array):
    """
    Returns a dictionary containing a list of conditional probabilities 
    (as logs) for each word for each given the categories and the probability 
    for each category as logs
    """
    # Calculate prior for each category     
    p_restaurants = math.log(categories.count('Restaurants')/len(categories))
    p_nightlife = math.log(categories.count('Nightlife')/len(categories))
    p_shopping = math.log(categories.count('Shopping')/len(categories))


    # Aggregate all CONDITIONAL probabilites of review words
    # for each given category
    rest_total = np.ones(shape=rv_array[0].shape, dtype=np.int64)
    night_total = np.ones(shape=rv_array[0].shape, dtype=np.int64)
    shop_total = np.ones(shape=rv_array[0].shape, dtype=np.int64)

    for i in range(len(rv_array)):
        if categories[i] == 'Restaurants':
            rest_total += rv_array[i] 
        elif categories[i] == 'Nightlife':
            night_total += rv_array[i]
        else:
            shop_total += rv_array[i]

    # Calculate conditional probabilities for each word in log form 
    rest_probs = []
    night_probs = []
    shop_probs = []
    for i in range(len(rest_total)):
        rest_probs.append(math.log(rest_total[i] / len(rest_total)))
        night_probs.append(math.log(night_total[i] / len(night_total)))
        shop_probs.append(math.log(shop_total[i]/ len(shop_total)))
    
    return {
        'restaurant': (p_restaurants, rest_probs),
        'nightlife': (p_restaurants, night_probs),
        'shopping': (p_shopping, shop_probs)
    }


# Calculate the POSTERIOR probability
def calc_post(train_file, test_file):
    cats = extract(train_file)['categories']
    reviews = extract(train_file)['reviews']
    probs = get_review_conds(cats, reviews)


    
    
