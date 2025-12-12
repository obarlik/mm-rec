
"""
Demo: LSH Training Dynamic
Shows how 'Training' (Gradient Descent) moves vectors into the same LSH bucket.
Unlike the previous demo, here we start with TOTAL RANDOMNESS.
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit

def demo_training():
    print("🚀 Starting Training Simulation...")
    print("Goal: Teach the model that 'Cat' and 'Dog' are similar, but 'Car' is different.")
    
    key = random.PRNGKey(42)
    dim = 64
    lr = 0.1
    
    # 1. Initialize RANDOM vectors (No pre-defined closeness!)
    # Cat and Dog are NOT close initially.
    embeddings = {
        "Cat": random.normal(random.PRNGKey(1), (dim,)),
        "Dog": random.normal(random.PRNGKey(2), (dim,)),
        "Car": random.normal(random.PRNGKey(3), (dim,)),
    }
    
    # Static LSH Projections (The Router stays fixed usually)
    hyperplanes = random.normal(random.PRNGKey(99), (dim, 4)) # 4 bits -> 16 buckets

    def get_bucket(v):
        proj = jnp.dot(v, hyperplanes)
        bits = (proj > 0).astype(jnp.int32)
        powers = 2 ** jnp.arange(4)
        return jnp.dot(bits, powers)

    print("\n🐣 Epoch 0 (Initialization - Random):")
    for word, vec in embeddings.items():
        print(f"  {word}: Bucket {get_bucket(vec)}")

    # 2. Define Loss Function (The "Teacher")
    # We want Cat and Dog to be close (Similarity), and Cat and Car to be far (Contrastive).
    def loss_fn(emb_cat, emb_dog, emb_car):
        # Euclidean Distances
        dist_cat_dog = jnp.sum((emb_cat - emb_dog)**2)
        dist_cat_car = jnp.sum((emb_cat - emb_car)**2)
        
        # Loss: Pull Cat-Dog together, Push Cat-Car apart
        # Margin loss: dist_pos - dist_neg + margin
        return dist_cat_dog - dist_cat_car

    # Gradients
    grad_fn = jit(grad(loss_fn, argnums=(0, 1, 2)))

    # 3. Training Loop
    print("\n🔄 Training... (Applying Gradients)")
    for epoch in range(1, 21):
        # Calculate gradients
        g_cat, g_dog, g_car = grad_fn(embeddings["Cat"], embeddings["Dog"], embeddings["Car"])
        
        # Update weights (SGD)
        embeddings["Cat"] -= lr * g_cat
        embeddings["Dog"] -= lr * g_dog
        embeddings["Car"] -= lr * g_car
        
        if epoch % 5 == 0:
            loss = loss_fn(embeddings["Cat"], embeddings["Dog"], embeddings["Car"])
            print(f"  Epoch {epoch}: Loss {loss:.4f}")

    print("\n🦅 Epoch 20 (After Training):")
    for word, vec in embeddings.items():
        print(f"  {word}: Bucket {get_bucket(vec)}")
        
    # Validation
    b_cat = get_bucket(embeddings["Cat"])
    b_dog = get_bucket(embeddings["Dog"])
    b_car = get_bucket(embeddings["Car"])
    
    if b_cat == b_dog and b_cat != b_car:
        print("\n✨ SUCCESS: 'Cat' and 'Dog' learned to share the same Expert (Bucket)!")
    else:
        print("\n⚠️ Note: Values moved correctly but maybe hash boundary wasn't crossed yet. Run longer.")

if __name__ == "__main__":
    demo_training()
