# Machine-Learning-Jupyter-Notebooks

Various Jupyter Notebooks written for learning core Machine Learning concepts

## Naive Bayes Classifier

Used the spambase datasets which each have 58 columns: the first 57 columns are input features, corresponding to different properties of an email, and the last column is an output label indicating spam (1) or non-spam (0). Notebook fits the Naive Bayes model using the training data, then reports the misclassification percentage (test error) of the Naive Bayes classifier.

## SVM and LDA

Implemented the soft-margin SVM using batch gradient descent and stochastic gradient descent, with a learning rate for j iterations as a(j) = n / (1 + j*n). Plotted the training accuracy over iterations using the digits_training/test_data/labels.csv files. Also implemented the soft-margin SVM with the RBF kernel, reported training and test accuracy, as well as 5 misclassified test images (still using digits datasets).

Implemented linear discriminant analysis, reported training and test accuracy, as well as 5 misclassified test images (still using digits datasets).

## Kernelized Ridge Regression

Applied the kernelized ridge regression to the steel ultimate tensile strengh dataset, and reported the root mean square error (RSME) of the following kernel and degrees: Polynomial with degrees 2, 3, 4; Gaussian kernel.

## K-Means Image Compression

Partitioned the image into blocks of size M × M and reshaped each block into a vector of length 3M^2, then wrote a program to cluster the vectores using the k-means algorithm.
Deliverables:
• A plot of the k-means objective function value versus iteration number.
• A side-by-side of the original and compressed images
• A picture of the difference of the two images, with a neutral gray (128, 128, 128) to the difference before generating the image.
• The relative mean absolute error of the compressed image

## EM Algorithm and HMM

Generated the data as follows:
• Use d = 1 and N = 500. Let x be sampled independently and uniformly from the interval [0, 1].
• Use π1 = 0.7, π2 = 0.3.
• Use w1 = −2, w2 = 1.
• Use b1 = 0.5, b2 = −0.5.
• Use σ1 = 0.4, σ2 = 0.3

Implemented EM algorithm and estimated model parameters with initial estimates as:
• Use ˆπ1 = ˆπ2 = 0.5.
• Use ˆw1 = 1, ˆw2 = −1.
• Use ˆb1 = ˆb2 = 0.
• Use ˆσ1 = ˆσ2 = standardDev(y).

Deliverables:
• A plot of the (marginal) log-likelihood as a function of iteration number. Terminated when the log-likelihood increases by less than 10^−4.
• The estimated model parameters.
• A plot showing the data and estimated lines together.

Given the folllowing HMM: A = [[0.5, 0.2, 0.3], [0.2, 0.4, 0.4], [0.4, 0.1, 0.5]] φ = [[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]] π0 = [0.5, 0.3, 0.2], and a sequence of observations 0101, computed the posterior distribution over the sequence of states and reported the 3 most probable sequences

Sampled 5000 observation sequences of length 4 from the HMM. Then, treated the first N sequences as training data and learned the HMM parameters by the Baum–Welch algorithm.
Referred to exercise 13.12 (page 648-649) in Bishop’s book “Pattern Recognition And Machine Learning” for the E step and M step details.
Ran EM for 50 iterations and after each iteration, computed the unconditional distributions over all possible observation sequences of length 4 given by the current parameters and compared to the distribution given by the true parameters.
Plotted the KL divergence over each iteration for each N sequences used (500/1000/2000/5000)
