import csv
import math
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Filenames
all_training = "./dataset/train.csv"
all_test = "./dataset/test.csv"
self_training = "./dataset/self-train.csv"
self_test = "./dataset/self-test.csv"
predict_output = "./dataset/predictions.csv"

# Data preprocessing
def extract(filename):
    """
    Extracts reviews as a vectorized np array, categories
    depending on the data and words
    """
    reviews = []
    review_ids = []
    all_categories = []

    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            reviews.append(row['review'])
            training = len(row) == 7 
            if training: 
                all_categories.append(row['category'])
            else:
                review_ids.append(row['ID'])

    vectorizer = CountVectorizer()
    rv_vectors = vectorizer.fit_transform(reviews)
    words = vectorizer.get_feature_names_out() 
    reviews_array = rv_vectors.toarray()
    return {
        'categories': all_categories,
        'reviews': reviews_array,
        'words': words,
        'ids': review_ids
    }


# Calculate probabilities
def get_review_conds(categories, rv_array, words):
    """
    Returns a dictionary containing a list of conditional probabilities 
    (as logs) for each word, for each given category, and the probability 
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
        # len(rest_total) cancels itself out with np.ones initialization
        rest_probs.append(math.log(rest_total[i] / np.sum(rest_total))) 
        night_probs.append(math.log(night_total[i] / np.sum(night_total)))
        shop_probs.append(math.log(shop_total[i]/ np.sum(shop_total)))

    # Build word and probability dictionary
    rest_probs_dict = {words[i]:rest_probs[i] for i in range(len(words))}
    night_probs_dict = {words[i]:night_probs[i] for i in range(len(words))}
    shop_probs_dict = {words[i]:shop_probs[i] for i in range(len(words))}

    return {
        'restaurant': (p_restaurants, rest_probs_dict),
        'nightlife': (p_nightlife, night_probs_dict),
        'shopping': (p_shopping, shop_probs_dict)
    }


# Calculate the POSTERIOR probability
def calc_post(train_file, test_file):
    """
    Returns a dictionary of posterior probabilities for each
    row in the given test_file based on the training file
    """
    all_categories = extract(train_file)['categories']
    tr_reviews = extract(train_file)['reviews']
    tr_words = extract(train_file)['words']
    test_reviews = extract(test_file)['reviews']
    test_words = extract(test_file)['words']
    test_ids = extract(test_file)['ids']
    probs = get_review_conds(all_categories, tr_reviews, tr_words)
    rest_prior, night_prior, shop_prior = probs['restaurant'][0], probs['nightlife'][0], probs['shopping'][0]

    cat_predictions = []
    for row in test_reviews:
        rest_post_sum, night_post_sum, shop_post_sum = [rest_prior, 'Restaurants'], [night_prior, 'Nightlife'], [shop_prior, 'Shopping']
        for j in range(len(row)):
            test_word = test_words[j]
            if row[j] > 0 and test_word in tr_words:
                rest_post_sum[0] += row[j] * probs['restaurant'][1][test_word]
                night_post_sum[0] += row[j] * probs['nightlife'][1][test_word]
                shop_post_sum[0] += row[j] * probs['shopping'][1][test_word]
        cat_predictions.append(max(rest_post_sum, night_post_sum, shop_post_sum)[1])
    return {
        'ID': test_ids,
        'category': cat_predictions
    } 


def write_predictions(output_file, predictions_dict):
    df = pd.DataFrame(predictions_dict)
    df.to_csv(output_file, index=False)


# if __name__ == '__main__':
#     cat_predict = calc_post(all_training, all_test)
#     write_predictions(predict_output, cat_predict)

