#!/usr/bin/env python
# coding: utf-8

# - The following text reports the results of the analysis of the Udemy data set
# 
# - Udemy is a massive open online course (MOOC) platform that offers free and paid courses. Anyone can create a course, this business model allows Udemy to own hundreds of thousands of courses.

# # 1. Importing basic libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import empiricaldist
import datetime as dt
from sklearn.preprocessing import StandardScaler


# In[2]:


url = "https://raw.githubusercontent.com/al34n1x/DataScience/master/3.Pandas/udemy_courses_ejercicio.csv"


# In[3]:


df = pd.read_csv(url, sep = ";", comment = "#", parse_dates = ["published_timestamp"])


# Although the `comment = #` function can have NaNs, it can be used as an exercise to do some work on them.

# Observation of the first 5 rows of the dataset:

# In[4]:


df.head()


# Replacing the original dataset column names with more descriptive names:

# In[5]:


df.columns = ["course_id", "title", "url", "paid", "cost",
       "subscribers", "reviews", "lectures", "level",
       "duration", "date", "category"]


# Analysis of the data type of each column:

# In[6]:


df.dtypes


# In[7]:


print("The dataframe has: ", df.shape[0], "rows.")
print("The dataframe has: ", df.shape[1], "columns.")


# # 2. Identifying duplicate values

# In[8]:


df[df.duplicated(keep = False)].sort_values(by = "course_id")


# Being exact duplicates, they are eliminated and it is checked that said elimination has been carried out:

# In[9]:


df = df.drop_duplicates()


# In[10]:


df[df.duplicated(keep = False)]


# # 3. Identification and imputation of NaNs

# In[11]:


df.isna().sum()


# ## a.  Analysis of amount of NaNs: column "url"

# In[12]:


mask_nula = df["url"].isna()
df[mask_nula]


# In the case of the `url` column, we proceed to eliminate all the rows since we do not have data:

# In[13]:


df = df.dropna(subset = "url")
df.isna().sum()


# ## b.  Analysis of amount of NaNs: column "level"

# As there are still 31 nulls left, we try to impute them so as not to lose more data:

# In[14]:


df["level"].value_counts(normalize = True).round(2)


# Observing that the mode is "All Levels", the NaNs are replaced by "All Levels":

# In[15]:


df["level"] = df["level"].fillna("All Levels")
df.isna().sum()


# ## c. Analysis of amount of NaNs: column "cost"

# In[16]:


df.loc[df["cost"].isnull(), ["title", "category", "level", "paid", "cost"]]


# Note that all courses with `cost = NaN` correspond to free courses, since `paid level == False`.
# It is decided to replace the Nan by 0:

# In[17]:


df["cost"] = df["cost"].fillna(0)
df.isna().sum()


# # 4. Creating variables "gain" and "year"

# In[18]:


df["year"] = df["date"].dt.year.astype("int")


# In[19]:


df["gain"] = df["cost"] * df["subscribers"]


# # 5. Descriptive analysis

# ## A. General analysis

# In[20]:


df.describe().drop(["course_id", "lectures"], axis = 1).applymap(lambda x: f"{x:0.2f}")


# - It is observed that the average cost of the courses is 66.16 USD and the median is 45 USD.
# - The average duration of the courses is 4 hours.
# - Most of the courses belong to the year 2016.
# - There are a total of 268,923 subscribers

# It is noteworthy that the minimum number of subscribers is 0.

# In[21]:


mask_subscribers_zero = df["subscribers"] == 0
subscribers_zero = df[mask_subscribers_zero]
print("Total number of courses without subscribers: ", len(subscribers_zero))
subscribers_zero.head()


# It is possible to analyze which is the category and the level that predominates when `subscribers = 0`:

# In[22]:


subscribers_zero_cat = subscribers_zero["category"].value_counts(normalize = True).sort_values().round(2) * 100
subscribers_zero_cat


# It is observed that no course in the `category = Web Development` has 0 subscribers and that more than 50% of the courses with `subscribers = 0` belong to the `category = Business Finance`.

# The graph of the observation made is made:

# In[23]:


plt.style.use("seaborn-whitegrid")
fig, ax = plt.subplots(figsize = (8, 6))

ax.bar(subscribers_zero_cat.index, subscribers_zero_cat, color = "blue")

plt.xlabel("\n Categories", fontsize = 18)
plt.ylabel("Proportion of courses by category (%)", fontsize = 18)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18, rotation = 45)
plt.title("Number of courses with 0 subscribers", fontsize = 20)

plt.legend(["Total number of courses \n without subscribers: 65"], fontsize = 18, loc = 2)

plt.show()


# To perform the same analysis on variables with `type = "object"` you have to change the data type, from `object` to `category`.

# In[24]:


df[["category", "paid", "level"]] = df[["category", "paid", "level"]].astype("category")


# With the `unique` row you can see the number of categories per column.
# 
# With the `top` row you can see the category that is repeated the most.
# 
# With the `freq` row you can see the number of occurrences of that category.

# In[25]:


df["year"].value_counts(normalize = True)


# In[26]:


categoricals = df[["category", "paid", "level"]].describe()
categoricals.index = ["No. of observations", "Unique values", "Most repeated value", "Frequency"]
categoricals.columns = ["Course Category", "Is it payment?", "Course Level"]
categoricals


# - The most predominant category is `Web Development`.
# - Most courses are paid.
# - The most predominant level is `All Levels`.

# - A pivot table is created with MultiIndex `year`, `category`.
# - The number of courses in total ["course_id"].count() is analyzed.
# - The maximums for each category are highlighted in blue.

# A comparison is made between the number of subscribers to paid courses and free courses `(Paid = False)`.

# In[27]:


subscribers_paid = df.groupby("paid")["subscribers"].sum()
subscribers_paid = (subscribers_paid / df["subscribers"].sum()) * 100
print("Proportion of paid courses: {:.2f}%".format(subscribers_paid[True]))


# In[28]:


plt.style.use("ggplot")
fig, ax = plt.subplots(figsize = (8, 6))

ax.bar(["Free", "Paid"], [30.61, 69.38], color = "blue")

plt.ylim((0, 80))
plt.axhline(69.38, color = "red", linestyle = "--", xmax = 0.95)

plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
plt.xlabel("\n Course cost", fontsize = 18)
plt.ylabel("Subscriber ratio (%)", fontsize = 18)
plt.title("Proportion of subscribers per course cost", fontsize = 20)

plt.show()


# Analysis of the distribution between the number of subscribers and the cost of the course.

# In[29]:


plt.style.use("ggplot")
fig, ax = plt.subplots(figsize = (8, 6))

ax.scatter("cost", "subscribers", data = df, color = "blue")

plt.xlabel("Price (USD)", fontsize = 18)
plt.ylabel("Number of subscribers", fontsize = 18)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18)
plt.title("Relationship between price and number of subscribers", fontsize = 20)

plt.show()


# - It is observed that most of the courses belong to the free cost.
# - A course is observed that has more than 250 thousand subscribers.

# In[30]:


mask = df["subscribers"] > 200000
df[mask].loc[2827][["title", "cost", "category", "level", "subscribers", "duration"]]


# Analysis of the duration of the course and its cost:

# In[31]:


plt.style.use("ggplot")
fig, ax = plt.subplots(figsize = (8, 6))

ax.scatter("cost", "duration", data = df, color = "blue")

plt.xlabel("Price (USD)", fontsize = 18)
plt.ylabel("Duration", fontsize = 18)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18)
plt.title("Relationship between price and duration of the courses", fontsize = 20)

plt.show()


# The duration of the courses grows for those who are paid.

# Analysis between number of reviews and subscribers:

# In[32]:


plt.style.use("ggplot")
fig, ax = plt.subplots(figsize = (8, 6))

ax.scatter("subscribers", "reviews", data = df, color = "blue")

plt.xlabel("Number of subscribers", fontsize = 18)
plt.ylabel("Number of reviews", fontsize = 18)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18)
plt.title("Relationship between subscribers and number of reviews", fontsize = 20)

plt.show()


# A linear relationship is observed between the variables mentioned above.

# In[35]:


subscribers_time = df.pivot_table(values = "subscribers", index = "year")
subscribers_time = subscribers_time.fillna(0)
subscribers_time.style.highlight_max(color = "blue", axis = 0)


# In[37]:


plt.style.use("seaborn-whitegrid")

subscribers_time.plot(color = "steelblue", figsize = (8, 6))

plt.xlabel("\n Year", fontsize = 18)
plt.ylabel("Number of subscribers", fontsize = 18)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.title("Number of subscribers over time", fontsize = 20)
plt.legend([])

plt.show()


# The number of udemy subscribers drops as the years go by

# ## B. Category analysis

# The proportion of subscribers for each category is calculated:

# In[40]:


subscrib_category = (df.groupby("category")["subscribers"].count() / len(df)) * 100
subscrib_category = subscrib_category.sort_values().round(2)
subscrib_category


# In[41]:


plt.style.use("ggplot")
fig, ax = plt.subplots(figsize = (8, 6))

ax.bar(subscrib_category.index, subscrib_category, color = "blue")

plt.ylim((0, 60))
plt.axhline(32.70, color = "red", linestyle = "--")
style = dict(size = 18, color = 'black', fontstyle = "oblique")
ax.text("Web Development", 32.70, "32.70%", ha = "center", va = "bottom", **style)

plt.xlabel("\n Categories per level", fontsize = 18)
plt.ylabel("Proportion of subscribers (%)", fontsize = 18)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18, rotation = 45)
plt.title("Proportion subscribers per level", fontsize = 20)

plt.show()


# In this case, `category = Business Finance` and `category = Web Development` represent 65% of the sample.

# Within each category, the proportion of subscribers of free courses `cost = 0` and with the maximum cost `cost = 200` is compared.

# In[42]:


subscrib_category_0 = df.loc[df["cost"] == 0, ["category", "subscribers"]]
subscrib_category_0 = (subscrib_category_0.groupby("category")["subscribers"].count() / len(subscrib_category_0)) *100
subscrib_category_0 = subscrib_category_0.sort_values().round(2)
subscrib_category_0


# In[43]:


subscrib_category_200 = df.loc[df["cost"] == 200, ["category", "subscribers"]]
subscrib_category_200 = (subscrib_category_200.groupby("category")["subscribers"].count() / len(subscrib_category_200)) *100
subscrib_category_200 = subscrib_category_200.sort_values().round(2)
subscrib_category_200


# In[44]:


plt.style.use("ggplot")
fig, ax = plt.subplots(figsize = (8, 6))

labels = ["Graphic Design", "Musical Instruments", "Business Finance", "Web Development"]

x = np.arange(len(labels)) 
width = 0.4

ax.bar(x - width/2, subscrib_category_0, color = "blue", width = 0.4, label = "Course Value: Free")
ax.bar(x + width/2, subscrib_category_200, color = "red", width = 0.4, label = "Course Value: 200 USD")

plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18, rotation = 45)
plt.ylim((0, 70))
plt.xticks(x, labels)

plt.xlabel("\n Categories", fontsize = 18)
plt.ylabel("Proportion of subscribers (%)", fontsize = 18)
plt.title("Proportion of subscribers per category", fontsize = 20)
plt.legend(fontsize = 18)

plt.show()


# - No free courses are observed for `categoria = Graphic Design`.
# 
# - For `Web Development` there is a higher proportion of subscribers who paid the maximum value.
# 
# - For `Graphic Design`, the trend just mentioned is reversed.

# In[45]:


subscribers_time_category = df.pivot_table(values = "subscribers", index = "year", columns = "category")
subscribers_time_category = subscribers_time_category.fillna(0)
subscribers_time_category.style.highlight_max(color = "blue", axis = 0)


# In[46]:


plt.style.use("seaborn-whitegrid")

subscribers_time_category.plot(color = ["blue", "red", "green", "black"], figsize = (8, 6))

plt.xlabel("\n Year", fontsize = 18)
plt.ylabel("Number of subscribers", fontsize = 18)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.title("Number of subscribers over time (by category)", fontsize = 20)
plt.legend(title = "Category: ", fontsize = 16, title_fontsize = 18)

plt.show()


# - All courses for different `category` show a drop in the number of subscribers.
# - It is observed that for the `categories = Business Finance`, `categories = Graphic Design`, `categories = Musical Instruments` there are no subscribers for the year 2011.
# - The most marked declines for the categories `Musical Instruments` and `Graphic Design` occur from the year 2012 - 2013.
# - For the `category = Web Development` from the year 2013 - 2017.

# In[47]:


courses_category_time = df.pivot_table(values = "course_id", index = "year", columns = "category", aggfunc = 'count')
courses_category_time = courses_category_time.fillna(0)
courses_category_time.style.highlight_max(color = 'blue', axis = 0)


# The year 2016 has the largest number of courses for each category

# In[48]:


plt.style.use("seaborn-whitegrid")

courses_category_time.plot(color = ["blue", "red", "orange", "black"], figsize = (8, 6))

plt.axvline(2016, color = "red", linestyle = "--", ymax = 0.95) 
plt.xlabel("\n Year", fontsize = 18)
plt.ylabel("Number of courses", fontsize = 18)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.title("Number of courses over time (according to category)", fontsize = 20)
plt.legend(title = "Type of course: ", fontsize = 16, title_fontsize = 18)

plt.show()


# - The graph shows an increase in the number of courses in `category = "Business Finance` as of 2012, flattening as of 2015.
# 
# - The graph shows a substantial increase in the number of courses in `category = "Web Development` between 2014 and 2015.
# 
# - For all categories, the number of courses decreases from 2016, a possible explanation could be the lack of data registration for such dates.

# - A groupby with MultiIndex `year, category` is created.
# - The average cost of each course is analyzed.
# - The maximum for each category is highlighted in blue.

# In[49]:


price_category = df.pivot_table(values = "cost", index = "year", columns = "category")
price_category = price_category.fillna(0)
price_category.style.highlight_max(color = 'blue', axis = 1)


# It can be seen that the `Business Finance and Web Development` categories have a higher average cost than `Graphic Design and Musical Instruments`.

# In[51]:


plt.style.use("seaborn-whitegrid")

price_category.plot(color = ["blue", "red", "orange", "black"], figsize = (8, 6))

plt.xlabel("\n Year", fontsize = 18)
plt.ylabel("Price (USD)", fontsize = 18)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.title("Cost of courses over time (according to category)", fontsize = 20)
plt.legend(title = "Category course: ", fontsize = 16, title_fontsize = 18)

plt.show()


# - It is observed that for the year 2011 the categories `Business Finance`, `Graphic Design` and `Musical Instruments` did not yet exist.
# 
# - `category = Web Development` and `category = Business Finance` present a higher average price in their courses

# ## C. Level analysis

# Cost analysis in terms of mean and median for each of the levels:

# In[112]:


all_levels = df.loc[df["level"] == "All Levels", "cost"]
beginner_levels = df.loc[df["level"] == "Beginner Level", "cost"]
intermediate_levels = df.loc[df["level"] == "Intermediate Level", "cost"]
expert_levels = df.loc[df["level"] == "Expert Level", "cost"]


# In[113]:


df_plot_levels = [all_levels,beginner_levels,intermediate_levels,expert_levels]


# In[114]:


plt.style.use("seaborn-whitegrid")
fig, ax = plt.subplots(figsize = (8, 6))

level_labels = ["All Levels", "Beginner Level", "Intermediate Level", "Expert Level"]
ax.boxplot(df_plot_levels, labels = level_labels)

plt.xlabel("\n Level", fontsize = 14)
plt.ylabel("Price", fontsize = 14)
plt.xticks(fontsize = 12)
plt.title("Relationship between type of level and price", fontsize = 16)
plt.axhline(df["cost"].mean(), color = "red", linestyle = "--") 

plt.show()


# - Beyond the outliers, it is observed that the cost for `level = Expert Level` is higher, although there do not seem to be significant differences.

# The proportion of subscribers for each level is calculated:

# In[82]:


subscrib_level = (df.groupby("level")["subscribers"].count() / len(df)) * 100
subscrib_level = subscrib_level.sort_values().round(2)
subscrib_level


# In[84]:


plt.style.use("ggplot")
fig, ax = plt.subplots(figsize = (8, 6))

ax.bar(subscrib_level.index, subscrib_level, color = "blue")

plt.ylim((0, 60))
plt.axhline(52.38, color = "red", linestyle = "--")
style = dict(size =18, color = 'black', fontstyle = "oblique")
ax.text('All Levels', 52.38, "52.38%", ha = "center", va = "bottom", **style)

plt.xlabel("\n Level", fontsize = 18)
plt.ylabel("Proportion of subscribers (%)", fontsize = 18)
plt.yticks(fontsize = 18)
plt.xticks(fontsize = 18, rotation = 45)
plt.title("Proportion of subscribers per level", fontsize = 20)

plt.show()


# The levels `All Levels, Beginner Level` represent almost 90% of the sample.

# The analysis is deepened, looking for the proportion of subscribers by `level`, according to no `cost = 0` (minimum price that a course can have) or `cost = 200` (maximum price that a course can have).

# In[85]:


subscrib_level_0 = df.loc[df["cost"] == 0, ["level", "subscribers"]]
subscrib_level_0 = (subscrib_level_0.groupby("level")["subscribers"].count() / len(subscrib_level_0)) *100
subscrib_level_0 = subscrib_level_0.sort_values().round(2)
subscrib_level_0


# In[86]:


subscrib_level_200 = df.loc[df["cost"] == 200, ["level", "subscribers"]]
subscrib_level_200 = (subscrib_level_200.groupby("level")["subscribers"].count() / len(subscrib_level_200)) *100
subscrib_level_200 = subscrib_level_200.sort_values().round(2)
subscrib_level_200


# In[88]:


plt.style.use("ggplot")
fig, ax = plt.subplots(figsize = (8, 6))

labels = ["Expert Level", "Intermediate Level", "Beginner Level", "All Levels"]

x = np.arange(len(labels)) 
width = 0.4

ax.bar(x - width/2, subscrib_level_0, color = "blue", width = 0.4, label = "Course Value: Free")
ax.bar(x + width/2, subscrib_level_200, color = "red", width = 0.4, label = "Course Value: 200 USD")

plt.xticks(fontsize = 18, rotation = 45)
plt.yticks(fontsize = 18)
plt.ylim((0, 70))
plt.xticks(x, labels)

plt.xlabel("\n Level", fontsize = 18)
plt.ylabel("Proportion of subscribers (%)", fontsize = 18)
plt.title("Proportion of subscribers per level", fontsize = 20)
plt.legend(fontsize = 18)

plt.show()


# - For `Expert Level` no free courses are observed.
# 
# - For `All Levels` there is a higher proportion of subscribers who paid the maximum value.
# 
# - For `Beginner Level`, the trend just mentioned is reversed.

# In[108]:


subscribers_time_level = df.pivot_table(values = "subscribers", index = "year", columns = "level")
subscribers_time_level = subscribers_time_level.fillna(0)
subscribers_time_level.style.highlight_max(color = 'blue', axis = 0)


# In[111]:


plt.style.use("seaborn-whitegrid")

subscribers_time_level.plot(color = ["blue", "red", "green", "black"], figsize = (8, 6))

plt.xlabel("\n Year", fontsize = 18)
plt.ylabel("Number of subscribers", fontsize = 18)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.title("Number of subscribers over time (by level)", fontsize = 20)
plt.legend(title = "Level: ", fontsize = 16, title_fontsize = 18)

plt.show()


# - All courses for different `category` show a drop in the number of subscribers.
# - It is observed that for the `categories = Business Finance`, `categories = Graphic Design`, `categories = Musical Instruments` there are no subscribers for the year 2011.
# - The most marked declines for the categories `Musical Instruments` and `Graphic Design` occur from the year 2012 - 2013.
# - For the `category = Web Development` from the year 2013 - 2017.

# - A pivot table with MultiIndex `anio`, `nivel` is created.
# - The number of courses in total `["id_curso"].count()` is analyzed.
# - The maximum for each level is recorded in blue.

# In[92]:


course_level_time = df.pivot_table(values = "course_id", index = "year", columns = "level", aggfunc = "count")
course_level_time = course_level_time.fillna(0)
course_level_time.style.highlight_max(color = 'blue', axis = 0)


# In[93]:


plt.style.use("seaborn-whitegrid")

course_level_time.plot(color = ["blue", "red", "orange", "black"], figsize = (8, 6))

plt.axvline(2016, color = "red", linestyle = "--", ymax = 0.95) 
plt.xlabel("\n Año", fontsize = 18)
plt.ylabel("Number of courses", fontsize = 18)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.title("Number of courses over time (according to level)", fontsize = 20)
plt.legend(title = "Level course: ", fontsize = 16, title_fontsize = 18)

plt.show()


# - The graph shows a greater growth of the `All Levels` and `Beginner Level` levels.
# 
# - Again for 3 of 4 levels, the number of courses decreases from 2016.

# - A groupby with MultiIndex `year, level` is created.
# - The average cost of each level is analyzed.
# - The maximum for each level is highlighted in blue.

# In[97]:


price_level = df.pivot_table(values = "cost", index = "year", columns = "level")
price_level = price_level.fillna(0)
price_level.style.highlight_max(color = 'blue', axis = 0)


# As expected, the `Expert Level` courses are the most expensive. In turn, since 2012, an increase in the average cost of courses has been observed for most levels.

# In[100]:


plt.style.use("seaborn-whitegrid")

price_level.plot(color = ["blue", "red", "orange", "black"], figsize = (8, 6))

plt.xlabel("\n Year", fontsize = 18)
plt.ylabel("Price (USD)", fontsize = 18)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.title("Cost of courses over time (according to level)", fontsize = 20)
plt.legend(title = "Level course: ", fontsize = 16, title_fontsize = 18)

plt.show()


# - For all categories, the average price of the courses increases
# 
# - It is observed that only `level = All Levels` begins with a value greater than 0, for the year 2011. This indicates the possibility of the lack of categorization of courses according to level.
# 
# - `level = Expert Level` presents the highest average value

# ## D. What were the most profitable courses?

# In[115]:


df = df.sort_values(by = "gain", ascending=False)
df.head()


# In[117]:


top5 =  df[["title", "gain"]].sort_values(by = "gain", ascending = True).tail()
top5["gain"] = top5["gain"] / 1000
top5


# In[118]:


plt.style.use("seaborn-whitegrid")
fig, ax = plt.subplots(figsize = (8, 6))

ax.barh(top5["title"], top5["gain"], color = "blue")

plt.yticks(fontsize = 16)
plt.ylabel("Name course", fontsize = 20)
plt.xlabel("\n Profit expressed in thousands", fontsize = 20)
plt.xticks(fontsize = 16, rotation = 45)
plt.title("Gain Top 5 ", fontsize = 20)

plt.show()


# The predominance of courses dedicated to Web development can be observed.

# # 6. Conclusion

# - General:
#     - It is observed that the average cost of the courses was 66.16 USD and the median was 45 USD.
#     - The average duration of the courses was 4 hours.
#     - Most of the courses belong to the year 2016.
#     - There were a total of 268,923 subscribers and the number of udemy subscribers fell as the years go by.
#     - Most courses were paid (70%) and the duration of the courses grew for those who were paid.
#     - 4 of the 5 courses that gave udemy the most profit were `category = Web Development`.
#     
# - Category:
#     - The most predominant category was Web Development.
#     - `category = Web Development` and `category = Business Finance` presented a higher average price in their courses.
# 
# - Level:
#     - The most predominant level was `All Levels`.
#     - The price for `level = Expert Level` was higher compared to the other levels.
