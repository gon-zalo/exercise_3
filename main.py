import os
import sys
os.chdir(sys.path[0])

# necessary libraries
import fasttext
import fasttext.util
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import ttest_rel

fasttext.util.download_model('es', if_exists='ignore')  # downloads Spanish vectors if not available
fasttext_model = fasttext.load_model('cc.es.300.bin')  # loads embeddings

file_path = "dataset.csv"
df = pd.read_csv(file_path)

# calculate cosine similarities between pivot and inflection/derivation
results = []
for _, row in df.iterrows():
    pivot, inflection, derivation = row["pivot"], row["inflection"], row["derivation"]
    
    vec_pivot = fasttext_model.get_word_vector(pivot) # vector of the pivot
    vec_inflection = fasttext_model.get_word_vector(inflection) # vector of the inflection
    vec_derivation = fasttext_model.get_word_vector(derivation) # vector of the derivation

    # calculate similarity between pivot inflection/derivation
    # need 1 - cosine because cosine alone just measures distance, not similarity
    sim_inflection = 1 - cosine(vec_pivot, vec_inflection)
    sim_derivation = 1 - cosine(vec_pivot, vec_derivation)

    # calculate vector offsets
    offset_inflection = vec_inflection - vec_pivot
    offset_derivation = vec_derivation - vec_pivot

    # append everything to the list
    results.append((pivot, inflection, derivation, sim_inflection, sim_derivation, offset_inflection, offset_derivation))

# print results in a table format
print(f"{'Pivot':<15}{'Inflection':<15}{'Derivation':<15}{'P-I similarity':<15}{'P-D similarity':<15}")
for result in results:
    print(f"{result[0]:<15}{result[1]:<15}{result[2]:<15}{result[3]:<15.2f}{result[4]:<15.2f}")

# calculate cosine similarity between inflection and derivation offsets and prints it
similarity = 1 - cosine(offset_inflection, offset_derivation)
print(f"\nCosine Similarity between Inflection and Derivation Offsets: {similarity:.2f}")

# similarities for t-test
sim_inflection_values = [r[3] for r in results]
sim_derivation_values = [r[4] for r in results]

# convert to numpy array because scipy expects numpy arrays
sim_inflection_values = np.array(sim_inflection_values) 
sim_derivation_values = np.array(sim_derivation_values)

# calculate t-test
t_stat, p_value = ttest_rel(sim_inflection_values, sim_derivation_values)

# print t-test results
print(f"\nT-test result: t = {t_stat:.4f}, p = {p_value:.4f}")
if p_value < 0.05:
    print("Inflection is significantly more predictable than derivation!")
else:
    print("No significant difference found.")