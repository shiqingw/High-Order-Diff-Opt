from scipy.stats import chi2

n = 3  # Change this based on the dimension of x
probability = 1 - chi2.cdf(1/3, n)
print("Probability that Q > 1 is:", probability)
