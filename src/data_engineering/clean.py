import os
import re
from functools import reduce
from dotenv import load_dotenv
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, array_join, when, explode_outer, collect_list, concat_ws, coalesce, lit, lower, trim, regexp_replace, first
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, StringType

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load environment variables
load_dotenv()

# Initialize Spark session
spark: SparkSession = SparkSession.builder \
    .master(os.getenv("SPARK_URL", "local[*]")) \
    .appName("DataCleaningWithSpark") \
    .config("spark.ui.port", "4040") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

# Load directories from environment variables
base_data_dirs = [path.strip().strip('"') for path in os.getenv("BASE_DATA_DIR", "").split(",")]
base_cleaned_dir = os.getenv("BASE_CLEANED_DIR", "").strip().strip('"')
base_cleaned_file = os.getenv("BASE_CLEANED_FILE", "").strip().strip('"')

# Define the schema for input JSON files
schema = StructType([
    StructField("abstracts-retrieval-response", StructType([
        StructField("coredata", StructType([
            StructField("dc:title", StringType(), True),
            StructField("dc:description", StringType(), True)
        ]), True),
        StructField("authkeywords", StructType([
            StructField("author-keyword", ArrayType(
                StructType([StructField("$", StringType(), True)])
            ), True)
        ]), True),
        StructField("subject-areas", StructType([
            StructField("subject-area", ArrayType(
                StructType([StructField("$", StringType(), True)])
            ), True)
        ]), True)
    ]), True)
])

# Load and process files
dataframes = []

for data_dir in base_data_dirs:
    if not os.path.exists(data_dir):
        print(f"Warning: Directory {data_dir} does not exist. Skipping.")
        continue

    files = [os.path.join(root, file) for root, _, files in os.walk(data_dir) for file in files]
    print(f"Processing {len(files)} files in directory: {data_dir}")

    for file_path in files:
        try:
            # Read JSON file into Spark DataFrame
            df = spark.read.json(file_path, multiLine=True, schema=schema)

            # Exploding the 'author-keyword' first
            df_keywords = df.select(
                coalesce(col("abstracts-retrieval-response.coredata.dc:title"), lit("")).alias("title"),
                coalesce(col("abstracts-retrieval-response.coredata.dc:description"), lit("")).alias("description"),
                explode_outer(col("abstracts-retrieval-response.authkeywords.author-keyword")).alias("author_keyword_struct")
            )

            # Exploding the 'subject-area' second
            df_subjects = df.select(
                coalesce(col("abstracts-retrieval-response.coredata.dc:title"), lit("")).alias("title"),
                coalesce(col("abstracts-retrieval-response.coredata.dc:description"), lit("")).alias("description"),
                explode_outer(col("abstracts-retrieval-response.subject-areas.subject-area")).alias("subject_area_struct")
            )

            # Now, extract the string from the struct (i.e., access the "$" field for each keyword and subject area)
            df_keywords = df_keywords.withColumn("author_keyword", col("author_keyword_struct.$"))
            df_subjects = df_subjects.withColumn("subject_area", col("subject_area_struct.$"))

            # Filter out rows where subject_area is null or empty
            df_subjects = df_subjects.filter(col("subject_area").isNotNull() & (col("subject_area") != "") & col("subject_area").endswith("(all)"))

            # If no valid subject_area is found, skip the file
            if df_subjects.count() == 0:
                continue

            # Now, handle the case where subject-area is a single value
            df_subjects_single = df_subjects.groupBy("title", "description").agg(
                first("subject_area").alias("subject_area")
            )

            # Join with the df_keywords (processed keywords) DataFrame
            df_combined = df_keywords.join(df_subjects_single, on=["title", "description"], how="left_outer")

            # Continue with the remaining aggregation and cleaning...
            df_processed = df_combined.groupBy("title", "description").agg(
                concat_ws(" ", collect_list("author_keyword")).alias("keywords"),
                first("subject_area").alias("subject_area")
            )

            dataframes.append(df_processed)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# Combine all DataFrames into one
if dataframes:
    combined_df = reduce(lambda df1, df2: df1.unionByName(df2), dataframes)

  # Process and clean data
processed_df = combined_df.select(
    lower(trim(regexp_replace(col("title"), r"[^a-zA-Z\s]", ""))).alias("clean_title"),
    lower(trim(regexp_replace(col("description"), r"[^a-zA-Z\s]", ""))).alias("clean_abstract"),
    lower(trim(regexp_replace(col("keywords"), r"[^a-zA-Z\s]", ""))).alias("clean_keywords"),
    trim(col("subject_area")).alias("clean_subject_area")
)

# Convert to a list of dictionaries
data = processed_df.toPandas().to_dict(orient="records")

# Save as compact JSON
# create the directory if it does not exist
os.makedirs(base_cleaned_dir, exist_ok=True)
with open(os.path.join(base_cleaned_dir, base_cleaned_file), "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

# Stop Spark session
spark.stop()