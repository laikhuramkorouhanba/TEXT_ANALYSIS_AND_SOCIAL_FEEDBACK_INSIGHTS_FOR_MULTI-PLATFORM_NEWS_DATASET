

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, lower, regexp_replace, avg, count, window, to_date
from pyspark.sql.functions import date_format, avg, sum as _sum
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

# Initialize Spark Session
spark = SparkSession.builder.appName("NewsDataAnalysis").getOrCreate()

# Load data
file_path = "/home/Korouhanba/Downloads/HW02/News_Final.csv"
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Clean text: Lowercase and remove punctuation
def clean_text(column):
    return regexp_replace(lower(col(column)), "[^a-z0-9\s]", "")

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# format date data-type
filtered_data = data.withColumn(
    "PublishDate",
    to_date(col("PublishDate"), "yyyy-MM-dd")
)

filtered_data.select("PublishDate").show(10, truncate=False)

# Convert PublishDate to DateType
final_filtered_data = filtered_data.withColumn("PublishDate", F.to_date("PublishDate", "yyyy-MM-dd"))

# Filter out rows with NULL or invalid PublishDate
final_filtered_data = final_filtered_data.filter(F.col("PublishDate").isNotNull())
final_filtered_data.select("PublishDate").show(10, truncate=False)

# List of valid topics
valid_topics = ['economy', 'obama', 'microsoft', 'palestine']

# Filter the data to remove rows with invalid topics
cleaned_data = data.filter(data.Topic.isin(valid_topics))

# Subtask (1): Word Count
def word_count(df, column):
    words = df.withColumn("word", explode(split(clean_text(column), "\s+")))
    word_freq = words.groupBy("word").count().orderBy(col("count").desc())
    return word_freq

title_word_count = word_count(data, "Title")
headline_word_count = word_count(data, "Headline")

print("Total Word Count for Titles:")
title_word_count.show(10)
print("Total Word Count for Headlines:")
headline_word_count.show(10)


# Word count per day for both Title and Headline
words_per_day_title = (
    final_filtered_data.withColumn("word", explode(split(clean_text("Title"), "\s+")))
    .groupBy("word", "PublishDate")
    .count()
    .orderBy("PublishDate", col("count").desc())
)

words_per_day_headline = (
    final_filtered_data.withColumn("word", explode(split(clean_text("Headline"), "\s+")))
    .groupBy("word", "PublishDate")
    .count()
    .orderBy("PublishDate", col("count").desc())
)

# Word count per topic for both Title and Headline
words_per_topic_title = (
    cleaned_data.withColumn("word", explode(split(clean_text("Title"), "\s+")))
    .groupBy("word", "Topic")
    .count()
    .orderBy("Topic", col("count").desc())
)

words_per_topic_headline = (
    cleaned_data.withColumn("word", explode(split(clean_text("Headline"), "\s+")))
    .groupBy("word", "Topic")
    .count()
    .orderBy("Topic", col("count").desc())
)

# print the word count per day and topic for Titles and Headlines
print("Word Count per Day (Title):")
words_per_day_title.show(10)
print("Word Count per Day (Headline):")
words_per_day_headline.show(10)
print("Word Count per Topic (Title):")
words_per_topic_title.show(10)
print("Word Count per Topic (Headline):")
words_per_topic_headline.show(10)

# SubTask (2): Popularity Calculation

# Convert the PublishDate to Hour and Day
df = data.withColumn('hour', date_format('PublishDate', 'yyyy-MM-dd HH')) \
       .withColumn('day', date_format('PublishDate', 'yyyy-MM-dd'))

# Calculate average popularity by hour and by day for each platform
popularity_by_hour = df.groupBy('hour').agg(
    avg('Facebook').alias('avg_Facebook'),
    avg('GooglePlus').alias('avg_GooglePlus'),
    avg('LinkedIn').alias('avg_LinkedIn')
)

popularity_by_day = df.groupBy('day').agg(
    avg('Facebook').alias('avg_Facebook'),
    avg('GooglePlus').alias('avg_GooglePlus'),
    avg('LinkedIn').alias('avg_LinkedIn')
)

popularity_by_hour.show(10)
popularity_by_day.show(10)

# Hour - avg_Facebook
popularity_by_hour.select("hour", "avg_Facebook").write.csv("/home/Korouhanba/Downloads/HW02/hour_avg_facebook.csv", header=True)

# Hour - avg_GooglePlus
popularity_by_hour.select("hour", "avg_GooglePlus").write.csv("/home/Korouhanba/Downloads/HW02/hour_avg_googleplus.csv", header=True)

# Hour - avg_LinkedIn
popularity_by_hour.select("hour", "avg_LinkedIn").write.csv("/home/Korouhanba/Downloads/HW02/hour_avg_linkedin.csv", header=True)

# Day - avg_Facebook
popularity_by_day.select("day", "avg_Facebook").write.csv("/home/Korouhanba/Downloads/HW02/day_avg_facebook.csv", header=True)

# Day - avg_GooglePlus
popularity_by_day.select("day", "avg_GooglePlus").write.csv("/home/Korouhanba/Downloads/HW02/day_avg_googleplus.csv", header=True)

# Day - avg_LinkedIn
popularity_by_day.select("day", "avg_LinkedIn").write.csv("/home/Korouhanba/Downloads/HW02/day_avg_linkedin.csv", header=True)

# SubTask (3): Sentiment Calculation

# List of topics to analyze
topics = ['economy', 'microsoft', 'obama', 'palestine']

# Filter and calculate sum of sentiment score for each topic
sentiment_by_topic = (
    data.filter(col('Topic').isin(topics))  # Filter for specific topics
    .groupBy('Topic')  # Group by topic
    .agg(
        (_sum('SentimentTitle') + _sum('SentimentHeadline')).alias('sum'),
        ((_sum('SentimentTitle') + _sum('SentimentHeadline')) / 2).alias('sentiment_score')
    )
)

# Show the results
sentiment_by_topic.show()
