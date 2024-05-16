from scipy.stats import norm

# Define the percentage of data you want to cover (e.g., 80%)
percentage = 0.80

# Find the value of x such that the cumulative distribution function (CDF) at (1 + x) standard deviations is equal to the desired percentage
x = norm.ppf(percentage + (1-percentage)/2) - 1  # Adding (1-percentage)/2 to center the range around the mean
# Total standard deviations needed
total_std_dev = 1 + x

print("Total standard deviations needed to gather {}% of the data:".format(percentage * 100), total_std_dev)
