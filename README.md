# Using-Neural-Nets-for-the-TMBD-Box-Office-Prediction-Competition

Keywords: Kaggle, TMDB Box Office Prediction, Neural Nets, Keras, Dropout, Regularization, Scaling, One-hot encoding Sci-kit learn, Random Forest

This python script uses the same data set, for the TMDB box office prediction in R, and the final data set used by the neural net was created
in R. Here instead of using a stacked model to predict box office revenues (like in R) a neural net is created with 3 hidden layers containing 256,128 and 64
activation units utilizng the ReLu activation unit. To prevent overfitting both dropout and regularization are utulized. The dropout is used 
for each hidden layer and the probability a unit is dropped in 0.35. Each layers also uses L2 regularization with a penalty parameter of 0.01.
The neural net is implemented using Keras. While the data used has already been pre-processed, the factor variables are converted into numeric variables
using one-hot encode using the pandas method get_dummies(). Scaling the continous variables is also done using the sci-kit learn function 
StandarScaler(), which normalizes the continous variables to their Z scores(so all the variables have roughly the same scale) so that convergence 
of the optimzation algorithm occurs more quickly. 

In this script a random forest method is also used to predict on the same data. The random forest method is implmented through sci-kit learn
function RandomForestRegressor() and the optimal hyperparamters (minimum number of values in a leaf, and the number of variables used at each split) are found using the 
sci-kit learn function GridSearchCV(). Neither the random forest in sci-kit learn or the neural net outperformed the models in R. Although the neural
net had the lowest cross validated error.

The data set used in this script is contained in a zip file in this repository. For further infromation on the data set and how it was processed see
the repositiory Kaggle Competition TMDB Box Office Prediction
