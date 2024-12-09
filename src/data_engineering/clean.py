import os
import re
from dotenv import load_dotenv
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, array_join, when, explode_outer, collect_list, concat_ws, coalesce, lit, lower, trim, regexp_replace
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, StringType, MapType 

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load environment variables from .env file
load_dotenv()

# Initialize Spark session using environment variables
spark: SparkSession = SparkSession.builder \
    .master(os.getenv("SPARK_URL", "local[*]")) \
    .appName("DataCleaningWithSpark") \
    .config("spark.ui.port", "4040") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "4") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

# Load environment variables
base_data_dirs = [path.strip().strip('"') for path in os.getenv("BASE_DATA_DIR", "").split(",")]
base_cleaned_dir = os.getenv("BASE_CLEANED_DIR", "").strip().strip('"')

# Define cleaning functions
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return ''.join(tokens)

def clean_keywords(keywords):
    if not isinstance(keywords, list):
        return []
    return list(set([kw for kw in keywords if kw not in ENGLISH_STOP_WORDS and len(kw) > 1]))

def clean_title(title):
    if not isinstance(title, str) or not title.strip():
        return ''
    title = title.title()
    common_words = ['a', 'an', 'the']
    title_words = title.split()
    if title_words and title_words[0].lower() in common_words:
        title = ' '.join(title_words[1:])
    return title

def clean_abstract(abstract):
    if not isinstance(abstract, str):
        return ''
    abstract = re.sub(r'[^\x00-\x7F]+', '', abstract)
    return abstract.strip()

# Register UDFs
clean_text_udf = udf(clean_text, StringType())
clean_keywords_udf = udf(clean_keywords, ArrayType(StringType()))
clean_title_udf = udf(clean_title, StringType())
clean_abstract_udf = udf(clean_abstract, StringType())

# Define the schema
schema = StructType([
    StructField("abstracts-retrieval-response", StructType([
        StructField("coredata", StructType([
            StructField("dc:title", StringType(), True),       # Title field, nullable
            StructField("dc:description", StringType(), True) # Description field, nullable
        ]), True),                                           # coredata struct, nullable
        StructField("authkeywords", StructType([            # authkeywords struct
            StructField("author-keyword", ArrayType(
                StructType([StructField("$", StringType(), True)])
            ), True)
        ]), True)
    ]), True)                                                # abstracts-retrieval-response struct, nullable
])

# Load and process files
dataframes = []

for data_dir in base_data_dirs:
    if not os.path.exists(data_dir):
        print(f"Warning: The directory {data_dir} does not exist.")
        continue

    # Iterate over all files in the directory
    # print how many files are in the directory
    print(f"Processing files in {data_dir} with {len(os.listdir(data_dir))} files")
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # print(f"Processing file: {file_path}")

            try:
                # Load data
                df = spark.read.json(file_path, multiLine=True, schema=schema)

                # df.show(truncate=False)
                # df.printSchema()

                # Process the data
                df_processed = df.select(
                    coalesce(col("abstracts-retrieval-response.coredata.dc:title"), lit("")).alias("title"),
                    coalesce(col("abstracts-retrieval-response.coredata.dc:description"), lit("")).alias("description"),
                    explode_outer(
                        col("abstracts-retrieval-response.authkeywords.author-keyword")
                    ).alias("author_keyword_struct")  # Explode first
                ).select(
                    col("title"),
                    col("description"),
                    col("author_keyword_struct.$").alias("author_keyword")  # Extract the "$" field
                ).groupBy("title", "description").agg(
                    concat_ws(" ", collect_list("author_keyword")).alias("keywords")  # Concatenate keywords with space
                )

                # # Show the result
                # df_processed.show(1)
                # df_processed.printSchema()

                # Append to the list of DataFrames
                dataframes.append(df_processed)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                
# Combine DataFrames safely
if len(dataframes) == 1:
    combined_df = dataframes[0]  # Only one DataFrame, no union needed
else:
    combined_df = dataframes[0]
    for df in dataframes[1:]:
        combined_df = combined_df.unionByName(df)

# Process and clean data
processed_df = combined_df.select(
    lower(trim(regexp_replace(col("title"), r"[^a-zA-Z\s]", ""))).alias("clean_title"),
    lower(trim(regexp_replace(col("description"), r"[^a-zA-Z\s]", ""))).alias("clean_abstract"),
    lower(trim(regexp_replace(col("keywords"), r"[^a-zA-Z\s]", ""))).alias("clean_keywords")
)

# Convert to a list of dictionaries
data = processed_df.toPandas().to_dict(orient="records")

# Save as compact JSON
# create the directory if it does not exist
os.makedirs(base_cleaned_dir, exist_ok=True)
with open(base_cleaned_dir + "/cleaned.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

# Stop Spark session
spark.stop()
