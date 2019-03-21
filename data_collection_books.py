#
# Books
#

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime

import findspark
findspark.init()
import pyspark

from pyspark.sql.types import DateType
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import *
import pyspark.sql.functions as fn
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.sql import SQLContext
 
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

DATA_DIR = 'data/'

### IMPORTING & CLEANING META DATA

meta_products = spark.read.json(DATA_DIR+"meta_Books.json")
# This will extract only the features and turn them into more readable features.
# Features removed : corruptRecord, imURL, related
# This will extract only the features and turn them into more readable features.
# Filter salesRank = None because this will lead to problems for the writing in parquet
# Features removed : corruptRecord, imURL, related
data_cleaned = meta_products.rdd.flatMap(lambda r: [(r[1], r[4], r[6], r[9] )]) \

# Define the StructType to define the DataFrame that we'll create with the previously extracted rdd table
schema = StructType([
    StructField("asin", StringType(), True),
    #StructField("brand", StringType(), True),
    #StructField("category", StringType(), True),
    StructField("description", StringType(), True),
    StructField("price", FloatType(), True),
    #StructField("salesRank", IntegerType(), True),
    StructField("title", StringType(), True)
])

# Transform the RDD data into DataFrame (we'll then be able to store it in Parquet)
datacleaned_DF = spark.createDataFrame(data_cleaned, schema=schema)

#Save into parquet to save time in the next times
datacleaned_DF.write.mode('overwrite').parquet("meta_Books.parquet")

# Read from the parquet data
#datacleaned_DF = spark.read.parquet("meta_HealthPersonalCare.parquet")
 
keywords = [" global warming", " solar energy", " recycling ", " pollution ", "solar power", " endangered species", "air pollution", \
" water pollution", " wind energy", " climate change", " wind power", " recycle ", " deforestation", " greenhouse effect", "environment", \
" sustainability ", " natural resources", "alternative energy", " climate ", "global warming", "renewable energy", " ecology", "composting", \
" carbon footprint", " bio ", " biosphere ", " renewable "]

# Filter with title and description not equal to None
# We will then be able to test if those features contains words defined in the keyword vector
# The keyword vector represents the thema that we want : ecology, bio etc...
filter_products_bio = datacleaned_DF.rdd.filter(lambda r: (r[3] != None) &  (r[1] != None)) \
                    .filter(lambda r: (any(word in r[3].lower() for word in keywords)) | (any(word in r[1].lower() for word in keywords)) ) 

# Transform the RDD data into DataFrame (we'll then be able work and join with review data)
DF_filter_products_bio = spark.createDataFrame(filter_products_bio, samplingRatio=0.2)

### IMPORTING & CLEANING REVIEWS DATA

reviews = spark.read.json(DATA_DIR+"reviews_Books.json")

#Save into parquet to save time in the next times
reviews.write.mode('overwrite').parquet("reviews_Books.parquet")

# Read from the parquet data
#reviews = spark.read.parquet("reviews_PatioLawnGarden.parquet")

### Join Reviews and Metadata
review_product_join = DF_filter_products_bio.join(reviews, ['asin'])

review_product_join.write.mode('overwrite').parquet("joined_Books.parquet")