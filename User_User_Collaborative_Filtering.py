# =========================================================
# For more info, see https://hoseinkh.github.io/projects/
# =========================================================
import pickle
import numpy as np
import pandas as pd
from sortedcontainers import SortedList
from tqdm import tqdm
## ********************************************************
## Parameters
num_neighbors_to_consider = 25 # number of neighbors we'd like to consider
min_num_of_common_movies_to_be_cosidered_similar = 5 # number of common movies users must have in common in order to consider
# we use this minimum to ensure the movies are similar enough to do the ...
# ... do the calculations. This helps to increase the accuracy of the model.
## ***************************
## Load dictionaries:
with open('./Data/user_to_movie.json', 'rb') as f:
  user_to_movie = pickle.load(f)
#
with open('./Data/movie_to_user.json', 'rb') as f:
  movie_to_user = pickle.load(f)
#
with open('./Data/user_and_movie_to_rating.json', 'rb') as f:
  user_and_movie_to_rating = pickle.load(f)
#
with open('./Data/user_and_movie_to_rating___test_data.json', 'rb') as f:
  user_and_movie_to_rating___test_data = pickle.load(f)
## ********************************************************
N_max_user_id_in_train = np.max(list(user_to_movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1_max_movie_id_in_train= np.max(list(movie_to_user.keys()))
m2_max_movie_id_in_test = np.max([m for (u, m), r in user_and_movie_to_rating___test_data.items()])
M_max_movie_id_in_tain_and_test = max(m1_max_movie_id_in_train, m2_max_movie_id_in_test) + 1
print("N:", N_max_user_id_in_train, "M:", M_max_movie_id_in_tain_and_test)
#
if N_max_user_id_in_train > 10000:
  print("N_max_user_id_in_train =", N_max_user_id_in_train, "are you sure you want to continue?")
  print("Comment out these lines if so...")
  exit()
#
#
## to find the user similarities, you have to do O(N^2 * M) calculations!
## in the "real-world" we would want to parallelize this
#
## note: we really only have to do half the calculations ...
# ... since w_ij is symmetric, however then we need to store them, ...
# ... hence here we simply sacrifice computational time for space. This ...
# ... trade-off depends on the implementation and the database.
neighbors = [] # store neighbors in this list. Neighbors of user i are neighbors[i]
averages = [] # each user's average rating for later use
deviations = [] # each user's deviation for later use
for i in tqdm(range(N_max_user_id_in_train)):
  ## For each user i: find the num_neighbors_to_consider closest users to user i
  movies_i = user_to_movie[i]
  movies_i_set = set(movies_i)
  # userMovie2ratings_dict___user_i
  # calculate avg and deviation
  userMovie2ratings_dict___user_i = { movie:user_and_movie_to_rating[(i, movie)] for movie in movies_i }
  avg_rating_for_user_i = np.mean(list(userMovie2ratings_dict___user_i.values()))
  ## let's calcualte the new ratings: rating_im - avg_i
  dev_of_rating__user_i = { movie:(rating - avg_rating_for_user_i) for movie, rating in userMovie2ratings_dict___user_i.items() }
  dev_of_rating__user_i_values = np.array(list(dev_of_rating__user_i.values()))
  sigma_i = np.sqrt(dev_of_rating__user_i_values.dot(dev_of_rating__user_i_values))
  #
  # save these for later use
  averages.append(avg_rating_for_user_i)
  deviations.append(dev_of_rating__user_i)
  #
  ## In the following we calculate the similarities between ...
  # ... other users with user i
  sl = SortedList()
  for j in range(N_max_user_id_in_train):
    # For each user j, we want to calculate the similarity
    # don't include user i
    if j != i:
      movies_j = user_to_movie[j]
      movies_j_set = set(movies_j)
      common_movies_Ui_Uj = (movies_i_set & movies_j_set) # intersection
      if len(common_movies_Ui_Uj) > min_num_of_common_movies_to_be_cosidered_similar:
        # this user has the minimum number of required common movies to be considered for the computations
        # calculate avg and deviation for this user
        userMovie2ratings_dict___user_j = { movie:user_and_movie_to_rating[(j, movie)] for movie in movies_j }
        avg_rating_for_user_j = np.mean(list(userMovie2ratings_dict___user_j.values()))
        dev_of_rating__user_j = { movie:(rating - avg_rating_for_user_j) for movie, rating in userMovie2ratings_dict___user_j.items() }
        dev_of_rating__user_j_values = np.array(list(dev_of_rating__user_j.values()))
        sigma_j = np.sqrt(dev_of_rating__user_j_values.dot(dev_of_rating__user_j_values))
        #
        ## calculate the correlation coefficient
        numerator = sum(dev_of_rating__user_i[m]*dev_of_rating__user_j[m] for m in common_movies_Ui_Uj)
        w_ij = numerator / (sigma_i * sigma_j)
        #
        # insert into sorted list and truncate
        # negate weight, because list is sorted ascending
        # maximum value (1) is "closest"
        # since we are interested in high values of the weights, we store -w_ij (later we remove the negative)
        sl.add((-w_ij, j))
        # we only need to consider the top neighbors, so ...
        # ... delete the last one if the size exceeds.
        if len(sl) > num_neighbors_to_consider:
          del sl[-1]
  #
  # store the top neighbors
  neighbors.append(sl)
  #
  # print out useful things
  # if i % 1 == 0:
  #   print(i)
#
## ********************************************************
## Make a prediction for the rating that user i gives to the movie m
def predict(i, m):
  if False: # we deactivate this for getting the training MSE
    ## Check to see if user i has already rated movie m or not ...
    # ... if so, return the actual rating!
    try:
      prediction = user_and_movie_to_rating[(i,m)]
      return prediction
    except KeyError:
      pass
  ## User i has not rated movie m. We need to predict it.
  # calculate the weighted sum of deviations
  numerator = 0
  denominator = 0
  for neg_w, j in neighbors[i]:
    # remember, the weight is stored as its negative
    # so the negative of the negative weight is the positive weight
    try:
      numerator += (-neg_w) * deviations[j][m]
      denominator += abs(neg_w)
    except KeyError:
      # neighbor may not have rated the same movie
      pass
  #
  if denominator == 0:
    # we can't do anything, hence use the user i's average rating as prediction
    prediction = averages[i]
  else:
    prediction = numerator / denominator + averages[i]
  #
  if True:
    ## The predicted rating can be anythong, but here for instance we want ...
    # ... to have the ratings between 0.5 and 5. Hence we curb it!
    # you can avoid this.
    prediction = min(5, prediction)
    prediction = max(0.5, prediction) # min rating is 0.5
  #
  return prediction
## ********************************************************
## ***************************
## using neighbors, calculate MSE for the train set
train_predictions = []
train_targets = []
for (i, m), target in user_and_movie_to_rating.items():
  # predict the rating that user i gives to movie m
  prediction = predict(i, m)
  #
  # save the prediction and target
  train_predictions.append(prediction)
  train_targets.append(target)
## ***************************
## using neighbors, calculate MSE for the test set
test_predictions = []
test_targets = []
for (i, m), target in user_and_movie_to_rating___test_data.items():
  # predict the rating that user i gives to movie m
  prediction = predict(i, m)
  #
  # save the prediction and target
  test_predictions.append(prediction)
  test_targets.append(target)
#
## ***************************
# calculate accuracy
def mse(pred, targ):
  pred = np.array(pred)
  targ = np.array(targ)
  return np.mean((pred - targ)**2)
#
#
print('MSE for train set = {}'.format(mse(train_predictions, train_targets)))
print('MSE for test set = {}'.format(mse(test_predictions, test_targets)))



