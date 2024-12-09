
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_model.predict.predict import *


# Define the configuration (ensure paths are correct)
config = {
    "model_save_path": "./models/LogisticRegression_single_hyper_01/2018_2023/single_label_classifier.pkl",
    "label_encoder_save_path": "./models/LogisticRegression_single_hyper_01/2018_2023/label_encoder.pkl",
    "tokenizer_model_save_dir": "./models/LogisticRegression_single_hyper_01/2018_2023/tokenizer_model/",
    "batch_size": 16,
    'tokenizer_model_name' : 'allenai/scibert_scivocab_uncased'
}

data = {
        'title': [
            "Utility-based cache partitioning: A low-overhead, high-performance, runtime mechanism to partition shared caches",
            "Utility-based cache partitioning: A low-overhead, high-performance, runtime mechanism to partition shared caches"

        ],
        'abstract': [
            "This paper investigates the problem of partitioning a shared cache between multiple concurrently executing applications. The commonly used LRU policy implicitly partitions a shared cache on a demand basis, giving more cache resources to the application that has a high demand and fewer cache resources to the application that has a low demand. However, a higher demand for cache resources does not always correlate with a higher performance from additional cache resources. It is beneficial for performance to invest cache resources in the application that benefits more from the cache resources rather than in the application that has more demand for the cache resources. This paper proposes utility-based cache partitioning (UCP), a low-overhead, runtime mechanism that partitions a shared cache between multiple applications depending on the reduction in cache misses that each application is likely to obtain for a given amount of cache resources. The proposed mechanism monitors each application at runtime using a novel, cost-effective, hardware circuit that requires less than 2kB of storage. The information collected by the monitoring circuits is used by a partitioning algorithm to decide the amount of cache resources allocated to each application. Our evaluation, with 20 multiprogrammed workloads, shows that UCP improves performance of a dual-core system by up to 23% and on average 11% over LRU-based cache partitioning",
            "-based cache partitioning (UCP), a low-overhead, runtime mechanism that partitions a shared cache between multiple applications depending on the reduction in cache misses that each application is likely to obtain for a given amount of cache resources. The proposed mechanism monitors each application at runtime using a novel, cost-effective, hardware circuit that requires less than 2kB of storage. The information collected by the monitoring circuits is used by a partitioning algorithm to decide the amount of cache resources allocated to each application. Our evaluation, with 20 multiprogrammed workloads, shows that UCP improves performance of a dual-core system by up to 23% and on average 11% over LRU-based cache partitioning",

        ],
        'keywords': [
            ["Runtime","Application software"],[]
        ]
    }

def pred():
    
    # Convert JSON data to DataFrame
    new_data_df = pd.DataFrame(data)
    
    # Make predictions
    predictions_df = predict_new_data(new_data_df, config)
    
    # Convert predictions to JSON
    result = predictions_df.to_dict(orient='records')[0]
    
    return result

print(pred())