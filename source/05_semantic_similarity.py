import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# === CONFIG ===
INPUT_CSV = "data/04_labeled/labeled_sentences.csv"
OUTPUT_DIR = "data/06_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV).dropna()
sentences = df["text"].tolist()
labels = df["label"].tolist()

# === COMPUTE EMBEDDINGS ===
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = model.encode(sentences, show_progress_bar=True)

# === COSINE SIMILARITY (Optional: inspect pairs) ===
sim_matrix = cosine_similarity(embeddings)
sim_df = pd.DataFrame(sim_matrix)
sim_df.to_csv(os.path.join(OUTPUT_DIR, "cosine_similarity_matrix.csv"), index=False)

# === VISUALIZATION ===
def visualize(method="pca"):
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        name = "tsne"
    else:
        reducer = PCA(n_components=2)
        name = "pca"

    reduced = reducer.fit_transform(embeddings)
    df_vis = pd.DataFrame(reduced, columns=["x", "y"])
    df_vis["label"] = labels

    plt.figure(figsize=(10,6))
    for label in sorted(set(labels)):
        subset = df_vis[df_vis.label == label]
        plt.scatter(subset.x, subset.y, label=f"Label {label}", alpha=0.6)

    plt.title(f"Sentence Embeddings ({name.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"embedding_{name}.png"))
    plt.show()

visualize("pca")
visualize("tsne")
