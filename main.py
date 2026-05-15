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
    Extracts reviews as a vectorized np array, categories
    depending on the data and headers
    """
    reviews = []
    all_categories = []

    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reviews.append(row['review'])
            training = len(row) == 7 
            if training: 
                all_categories.append(row['category'])

    vectorizer = CountVectorizer()
    rv_vectors = vectorizer.fit_transform(reviews)
    headers = vectorizer.get_feature_names_out() 
    reviews_array = rv_vectors.toarray()
    return {
        'categories': all_categories,
        'reviews': reviews_array,
        'headers': headers
    }


# Calculate probabilities
def get_review_conds(categories, rv_array, headers):
    """
    Returns a dictionary containing a list of conditional probabilities 
    (as logs) for each word for each given the categories and the probability 
    for each category as logs
    """
    # Calculate PRIOR for each category     
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

    # Build header and probability dictionary
    rest_probs_dict = {headers[i]:rest_probs[i] for i in range(len(headers))}
    shop_probs_dict = {headers[i]:rest_probs[i] for i in range(len(headers))}
    night_probs_dict = {headers[i]:rest_probs[i] for i in range(len(headers))}

    print(rest_probs_dict)
    
    return {
        'restaurant': (p_restaurants, rest_probs_dict),
        'nightlife': (p_restaurants, night_probs_dict),
        'shopping': (p_shopping, shop_probs_dict)
    }


# Calculate the POSTERIOR probability
def calc_post(train_file, test_file):
    priors = extract(train_file)['categories']
    tr_reviews = extract(train_file)['reviews']
    tr_headers = extract(train_file)['headers']
    test_reviews = extract(test_file)['reviews']
    probs = get_review_conds(priors, tr_reviews, tr_headers)

    # for row in test_reviews:
    #     for j in range(len(row)):
    #         if j == row.index('review'):
    #             print(True)

     

calc_post(all_training, all_test)
