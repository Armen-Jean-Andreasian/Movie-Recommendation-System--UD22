"""
Part 1

the link to deepnode project:
https://deepnote.com/workspace/name-cf55-45e1a18a-bc68-49bb-b4eb-f243e6d4ce79/project/Movier-Recommendation-System-b0b101d0-aebf-48bb-8ccc-faa3d2a1fee1/notebook/Machine%20Learning-ff1108a546914697bb9b02ed61701eb1

Project: Prediction of user rating based on collaborative based filtering
Source: ratings.csv
Libraries: scikit-surprise
"""



import pandas as pd
# pip install scikit-surprise
from surprise import Dataset, Reader
"""
The surprise library is primarily designed for building recommendation systems, where 
ratings are a common type of data used. 
... This is why it has a  rating_scale= operator
"""


def the_dataset():
    # Load the data
    ratings = pd.read_csv('ratings.csv')[['userId', 'movieId', 'rating']]


    # Create the dataset
    reader = Reader(rating_scale=(1, 5))  # the values are among 1-5 range
    # Dataset is being instantiated using .load_from_df() method
    dataset = Dataset.load_from_df(ratings, reader)
    return dataset


def the_trainset():
    dataset = the_dataset()

    # Build the trainset
    trainset = dataset.build_full_trainset()
    return trainset

