###Dealing with messy data###

##Find the missing values:a problem for the pipeline
print(df.info()) #get the not-null information
print(df.isnull()) #get True/False results
print(df['StackOverflowJobsRecommend'].isnull().sum())
#return
512

#finding non-missing values
print(df.notnull()) #get True/False results

##Dealing the missing value:
#listwise delection in Python--complete case analysis#按列表删除fully excluse in from your model if the value is missing
#Drop all rows with at least one missing values
df.dropna(how='any')

#Drop rows with missing values in a specific column
df.dropna(subset=['VersionControl'])

#we often check the shape before and after the dealing missing values operations
#the drawback of listwise  delection:delect perfectly vaild data points that sharing the row with missing values;depend on randomness;reduce the effective information,reduce the df(degree of freedom)

#Replacing with strings#
# Replace missing values in a specific column
# with a given string
df['VersionControl'].fillna(value='None Given', inplace=True)
# Record where the values are not missing
df['SalaryGiven'] = df['ConvertedSalary'].notnull() #get the True/False results
# Drop a specific column
df.drop(columns=['ConvertedSalary'])
 
#Print the count of occurrences,use .value_counts()
print(so_survey_df['Gender'].value_counts())

##Filling Continuous Missing Values## moving all rows which include at least one missing value
#for categorical columns:replacing missing values with the most common occuring value(mean,etc.) or with a string that missing value such as 'None'
#for numeric columns:replacing missng values with a suitable value
#measures of central tendency:mean, median
print(df['ConvertedSalary'].mean())
print(df['ConvertedSalary'].median())
#filling the missing values
df['ConvertedSalary'] = df['ConvertedSalary'].fillna(df['ConvertedSalary'].mean()) #filling the missing value with the mean
df['ConvertedSalary'] = df['ConvertedSalary']\.astype('int64')  #changing all the decimal values by changing data type to integar using astype() method
#rounding values
df['ConvertedSalary'] = df['ConvertedSalary'].fillna(
    round(df['ConvertedSalary'].mean()))

##Dealing with other data issue##
#bad character
print(df['RawSalary'].dtype)
print(df['RawSalary'].head())
#dealing with bad character
df['RawSalary'] = df['RawSalary'].str.replace(',', '') #remove all the comma notation
df['RawSalary'] = df['RawSalary'].astype('float')
#finding other stray character
coerced_vals = pd.to_numeric(df['RawSalary'],
                             errors='coerce')
print(df[coerced_vals.isna()].head()) #the result return the $ sign,we can use .replace() method to remove the the dollar signs

#chaining method
df['column_name'] = df['column_name'].method1()
df['column_name'] = df['column_name'].method2()
df['column_name'] = df['column_name'].method3()
#or:
df['column_name'] = df['column_name']\
                     .method1().method2().method3(). #one method after the another to get the result
