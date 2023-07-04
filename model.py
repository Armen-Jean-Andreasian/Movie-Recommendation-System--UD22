"""
Part 2

Prediction
Validation
"""
from surprise import SVD
from surprise import model_selection
from data_manipulations import the_trainset, the_dataset
"""
SVD: singular value decomposition. It's a module to construct matrix of users and items. 
"""

# creating and training an ML model

# instantiation
my_trainset = the_trainset()

svd = SVD()
svd.fit(trainset=my_trainset)


# Prediction
svd.predict(uid=15, iid=1956)
"""
>>> Prediction(uid=15, iid=1956, r_ui=None, est=3.2874221221107636, details={'was_impossible': False})
user ID 15
item ID 1956
r_ui : rating to handle manually if needed
est : is the PREDICTION
'was_impossible': False - could handle Prediction and get est value
----------
also another way is:
svd.predict(uid=15, iid=1956).est()  : which will give solely the est value
"""


# Validation

my_dataset = the_dataset()

model_selection.cross_validate(svd, my_dataset, measures=['RMSE', 'MAE'])

"""
RMSE stands for Root Mean Square Error
 MAE stands for Mean Absolute Error
"""


# output
"""
{'test_rmse': array([0.89867696, 0.90234267, 0.88746145, 0.89295101, 0.89794965]),
 'test_mae': array([0.69116011, 0.69566326, 0.68386239, 0.68827417, 0.68920116]),
 'fit_time': (1.0291883945465088,
  1.0885450839996338,
  1.0572445392608643,
  1.0768513679504395,
  1.0744099617004395),
 'test_time': (0.1608562469482422,
  0.15381622314453125,
  0.3196523189544678,
  0.14231133460998535,
  0.28181982040405273)}
"""

