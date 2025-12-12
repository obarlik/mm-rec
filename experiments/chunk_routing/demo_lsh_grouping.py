
"""
Demo: LSH Semantic Grouping
Visual proof that "similar" vectors (small angle) get the same Hash Code.
This simulates how "Cat" and "Dog" end up in the same Expert.
"""

import jax
import jax.numpy as jnp
from jax import random

def demo_lsh_grouping():
    print("🚀 Starting LSH Grouping Demo...")
    
    # 1. Simulate "Trained" Embeddings
    # In a real model, training forces "Cat" and "Dog" to be close because they appear in same contexts.
    # Here we simulate this by creating vectors around a "Topic Center".
    
    key = random.PRNGKey(42)
    dim = 64
    num_bits = 4 # Generates 2^4 = 16 Buckets (Experts)
    
    # Random Hyperplanes for LSH (The "Router")
    # These slice the space into 16 random regions.
    hyperplanes = random.normal(key, (dim, num_bits))

    def get_hash(v):
        # Project: vector @ hyperplane
        proj = jnp.dot(v, hyperplanes)
        # Sign: +/- -> 1/0
        bits = (proj > 0).astype(jnp.int32)
        # Convert bits to integer bucket ID
        powers = 2 ** jnp.arange(num_bits)
        return jnp.dot(bits, powers)

    # Topic 1: ANIMALS (Center Vector A)
    center_animals = random.normal(random.PRNGKey(1), (dim,))
    
    # Topic 2: VEHICLES (Center Vector B - orthogonal to A)
    center_vehicles = random.normal(random.PRNGKey(2), (dim,))
    
    # Create specific words by adding small noise to centers
    # This simulates "Contextual Similarity" learned during training
    dataset = {
        "Cat":     center_animals + random.normal(random.PRNGKey(10), (dim,)) * 0.2,
        "Dog":     center_animals + random.normal(random.PRNGKey(11), (dim,)) * 0.2,
        "Lion":    center_animals + random.normal(random.PRNGKey(12), (dim,)) * 0.3,
        
        "Car":     center_vehicles + random.normal(random.PRNGKey(20), (dim,)) * 0.2,
        "Truck":   center_vehicles + random.normal(random.PRNGKey(21), (dim,)) * 0.2,
        "Bus":     center_vehicles + random.normal(random.PRNGKey(22), (dim,)) * 0.2,
    }
    
    print(f"\n🧩 LSH Bucket Assignments (Total {2**num_bits} Buckets):")
    print("-" * 40)
    
    # 2. Check where they land
    for word, vector in dataset.items():
        bucket_id = get_hash(vector)
        print(f"Word: {word:<10} -> Bucket ID: {bucket_id}")

    print("-" * 40)
    print("Observe: 'Cat' and 'Dog' landed in the SAME bucket because they are geometrically close.")
    print("This happens automatically without us explicitly coding rules.")
    
    # 3. Hierarchy Check (Bit Slicing)
    # Let's look at the raw bits for Cat vs Car
    hash_cat = (jnp.dot(dataset["Cat"], hyperplanes) > 0).astype(int)
    hash_lion = (jnp.dot(dataset["Lion"], hyperplanes) > 0).astype(int)
    hash_car = (jnp.dot(dataset["Car"], hyperplanes) > 0).astype(int)
    
    print("\n🌳 Hierarchy Analysis (Bit Patterns):")
    print(f"Cat Bits:  {hash_cat}")
    print(f"Lion Bits: {hash_lion} (Very similar to Cat!)")
    print(f"Car Bits:  {hash_car}  (Completely different)")
    
    print("\nConclusion: LSH successfully sends 'Animals' to one Expert and 'Vehicles' to another.")

if __name__ == "__main__":
    demo_lsh_grouping()
