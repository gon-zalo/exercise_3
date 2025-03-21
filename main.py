import os
import sys
os.chdir(sys.path[0])

import fasttext
import fasttext.util
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import ttest_rel

fasttext.util.download_model('es', if_exists='ignore')  # Downloads Spanish vectors if not available
ft = fasttext.load_model('cc.es.300.bin')  # Load Spanish embeddings

file_path = "dataset.csv"  # Make sure dataset.csv is in the same directory as this script
df = pd.read_csv(file_path)

# Ensure the CSV has the correct column names
expected_columns = ["pivot", "inflection", "derivation"]
if not all(col in df.columns for col in expected_columns):
    raise ValueError(f"Dataset must have columns: {expected_columns}")

# Function to calculate vector offset
def vector_offset(pivot, modified_word, model):
    """Compute the vector difference between base and modified word."""
    vec_base = model.get_word_vector(pivot)
    vec_modified = model.get_word_vector(modified_word)
    return vec_modified - vec_base

# 3️⃣ Compute cosine similarities
results = []
for _, row in df.iterrows():
    pivot, inflection, derivation = row["pivot"], row["inflection"], row["derivation"]
    
    try:
        vec_pivot = ft.get_word_vector(pivot)
        vec_inflection = ft.get_word_vector(inflection)
        vec_derivation = ft.get_word_vector(derivation)

        sim_inflection = 1 - cosine(vec_pivot, vec_inflection)
        sim_derivation = 1 - cosine(vec_pivot, vec_derivation)

        offset_inflection = vec_inflection - vec_pivot  # Vector difference for inflection
        offset_derivation = vec_derivation - vec_pivot  # Vector difference for derivation

        results.append((pivot, inflection, derivation, sim_inflection, sim_derivation, offset_inflection, offset_derivation))

    except KeyError as e:
        print(f"Word not found: {e}")

# 4️⃣ Print results in a table format
print(f"{'Pivot':<12}{'Inflection':<12}{'Derivation':<12}{'Pivot-Inflection similarity':<20}{'Pivot-Derivation similarity':<20}")
for r in results:
    print(f"{r[0]:<12}{r[1]:<12}{r[2]:<12}{r[3]:<20.4f}{r[4]:<20.4f}")

print("Inflection Offset:", offset_inflection[:5])  # Print first 10 dimensions
print("Derivation Offset:", offset_derivation[:5])
similarity = 1 - cosine(offset_inflection, offset_derivation)
print(f"\nCosine Similarity between Inflection and Derivation Offsets: {similarity:.4f}")

# Extract similarity values from results
sim_inflection_values = [r[3] for r in results]  # Pivot-Inflection similarities
sim_derivation_values = [r[4] for r in results]  # Pivot-Derivation similarities

# convert to numpy array because scipy expects numpy arrays
sim_inflection_values = np.array(sim_inflection_values) 
sim_derivation_values = np.array(sim_derivation_values)

# Perform paired t-test
t_stat, p_value = ttest_rel(sim_inflection_values, sim_derivation_values)

print(f"\nT-test result: t = {t_stat:.4f}, p = {p_value:.4f}")
if p_value < 0.05:
    print("Inflection is significantly more predictable than derivation!")
else:
    print("No significant difference found.")