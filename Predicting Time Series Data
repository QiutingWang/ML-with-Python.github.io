###Predicting Time Series Data###

##Predicting data over time##
#Classification predict categorical outputs:classification_model.predict(X_test)
#Regression models predict continuous outputs:regression_model.predict(X_test)
#Correction of time series data will change overtime

#Visualizing relationships between time series:
fig, axs = plt.subplots(1, 2)
# Make a line plot for each timeseries
axs[0].plot(x, c='k', lw=3, alpha=.2)
axs[0].plot(y)
axs[0].set(xlabel='time', title='X values = time')
# Encode time as color in a scatterplot
axs[1].scatter(x_long, y_long, c=np.arange(len(x_long)), cmap='viridis')
axs[1].set(xlabel='x', ylabel='y', title='Color = time')
#fiting the regression model in sklearn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
model.predict(X)
#several different models fit on the same data
alphas = [.1, 1e2, 1e3]
ax.plot(y_test, color='k', alpha=.3, lw=3)
for ii, alpha in enumerate(alphas):  #we use ridge regression,alpha--the coefficient to be smoother and smaller,it is useful if you have noisy or correlated variables
    y_predicted = Ridge(alpha=alpha).fit(X_train, y_train).predict(X_test)
    ax.plot(y_predicted, c=cmap(ii / len(alphas)))
ax.legend(['True values', 'Model 1', 'Model 2', 'Model 3'])
ax.set(xlabel="Time")
#coefficient of determination(r^2):1-error(model)/variance(testdata)
from sklearn.metrics import r2_score
print(r2_score(y_predicted, y_test))

##Advanced time series prediction##
#data cleaning:for missing data and outliers
#using time to fill in missing data:interpolate missing values,using known values on either side of a gap in the data to make assumptions about what is missing
#Return a boolean that notes where missing values are,check the missing vlue
missing = prices.isna()
#count the missing values
missing_values = prices.isna().sum()
#Interpolate linearly within missing windows
prices_interp = prices.interpolate('linear') #we use the linear argument to fill in the blank;or we input:interpolation_type = 'zero'/'quadratic'(the missing value will be filled with a horizontal line) interpolate_and_plot(prices, interpolation_type)
#Plot the interpolated data in red and the data not missing values in black
ax = prices_interp.plot(c='r')
prices.plot(c='k', ax=ax, lw=2)

#using a rolling window to transform data#
#calculate each time piont percentage change over the mean of a window previous timepoints
#a common transformation is standardize its mean and variance over time
#Transform to % change with pd:
def percent_change(values):
    """Calculates the % change between the last value
    and the mean of previous values"""
    # Separate the last value and all previous values into variables
    previous_values = values[:-1]
    last_value = values[-1]
    # Calculate the % difference between the last value
    # and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) \
    / np.mean(previous_values)
    return percent_change
#Apply data using aggregate method
# Plot the raw data
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = prices.plot(ax=axs[0])
# Calculate % change and plot
ax = prices.rolling(window=20).aggregate(percent_change).plot(ax=axs[1])
ax.legend_.set_visible(False)

#use this transformation to detect outliers in our data:remove or replace outliers with a representative value
#outliers are out of 3 deviation from the mean usually
#plotting the threshold on our data:
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for data, ax in zip([prices, prices_perc_change], axs):
    # Calculate the mean / standard deviation for the data
    this_mean = data.mean()
    this_std = data.std()
    # Plot the data, with a window that is 3 standard deviations around the mean
    data.plot(ax=ax)
    ax.axhline(this_mean + this_std * 3, ls='--', c='r')
    ax.axhline(this_mean - this_std * 3, ls='--', c='r')
#if we have already transformed the raw data into % change,the outliers will be different,more outlier data points after transformation

#replacing the outliers using threshold,with the median of the remaining value#
#Center the data so the mean is 0
prices_outlier_centered = prices_outlier_perc - prices_outlier_perc.mean() #calculate the differences
#Calculate standard deviation
std = prices_outlier_perc.std()
#Use the absolute value of each datapoint to make it easier to find outliers
outliers = np.abs(prices_outlier_centered) > (std * 3)
#Replace outliers with the median value.We'll use np.nanmean since there may be nans around the outliers
prices_outlier_fixed = prices_outlier_centered.copy()
prices_outlier_fixed[outliers] = np.nanmedian(prices_outlier_fixed)
#visualize the results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
prices_outlier_centered.plot(ax=axs[0])
prices_outlier_fixed.plot(ax=axs[1]) #we remove the outliers

##Creating feature over time##
#extract features using window,define multiple functions of each window to extract many features at once
#Visualize the raw data
print(prices.head(3))
#Calculate a rolling window, then extract two features
feats = prices.rolling(20).aggregate([np.std, np.max]).dropna()
print(feats.head(3))
#use partial function when using dot-aggregate method
#If we just take the mean, it returns a single value
a = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
print(np.mean(a))
#We can use the partial function to initialize np.mean with an axis parameter
from functools import partial
mean_over_first_axis = partial(np.mean, axis=0) #the first argument is the function we want to modify
print(mean_over_first_axis(a))
#percentiles summarize your data,the N th percentile is the value where N% of the data is below that point,(100-N)% of data is above that point
print(np.percentile(np.linspace(0, 200), q=20)) #the first input is the percentile function takes an array
-->40
#combining np.percentile() with partial functions to calculate a range of percentiles
data = np.linspace(0, 100)
# Create a list of functions using a list comprehension
percentile_funcs = [partial(np.percentile, q=ii) for ii in [20, 40, 60]]
# Calculate the output of each function in the same way
percentiles = [i_func(data) for i_func in percentile_funcs]
print(percentiles)

#Calculating date-based features
#human feature date format:the days of week,holidays...
#Ensure our index is datetime
prices.index = pd.to_datetime(prices.index) #to_datetime() function to ensure dates treated as datetime object
-->Index([0 1 2 3 4 0 1 2 3 4], dtype='object')
#Extract datetime features
day_of_week_num = prices.index.weekday 
prices_perc['week_of_year'] = prices.index.week
prices_perc['month_of_year'] = prices.index.month
print(day_of_week_num[:10])
-->Index(['Monday' 'Tuesday' 'Wednesday' 'Thursday' 'Friday' 'Monday' 'Tuesday' 'Wednesday' 'Thursday' 'Friday'], dtype='object')
