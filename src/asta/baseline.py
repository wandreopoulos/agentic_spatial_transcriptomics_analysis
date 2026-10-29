import argparse
import glob
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import umap
import matplotlib.pyplot as plt
from scipy import sparse
import os
import time
from sklearn.metrics import silhouette_score
import seaborn as sns


class REClustering:
    def __init__(self, min_cluster_size=100, max_depth=3):
        self.min_cluster_size = min_cluster_size
        self.max_depth = max_depth

    def preprocess_data(self, expression_matrix):
        print("Starting preprocessing...")
        expression_matrix = expression_matrix.astype(np.float32)
        expression_matrix = np.log1p(expression_matrix)
        scaler = StandardScaler(copy=False)
        expression_matrix = scaler.fit_transform(expression_matrix)
        print("Preprocessing complete")
        return expression_matrix

    def multi_embedding(self, data):
        print("Generating embeddings...")
        embeddings = {}

        # PCA
        print("Computing PCA...")
        n_components = min(50, data.shape[1])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data)
        embeddings['pca'] = pca_result

        try:
            print("Computing UMAP...")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=max(15, min(data.shape[0] // 100, 50)),
                min_dist=0.1,
                metric='euclidean',
                random_state=42,
                low_memory=True,
                verbose=True
            )
            embeddings['umap'] = reducer.fit_transform(pca_result)

            print("Computing t-SNE...")
            tsne = TSNE(
                n_components=2,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            embeddings['tsne'] = tsne.fit_transform(pca_result)

        except Exception as e:
            print(f"Error in embedding generation: {str(e)}")

        return embeddings

    # Apply multiple clustering methods respectively
    def ensemble_clustering(self, embeddings, k):
        print("Performing multiple clustering methods...")
        results = {}

        for name, embedding in embeddings.items():
            try:
                # K-means
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                results[f'kmeans_{name}'] = kmeans.fit_predict(embedding)

                # DBSCAN
                eps = np.mean(np.std(embedding, axis=0)) * 0.5
                dbscan = DBSCAN(eps=eps, min_samples=5, n_jobs=-1)
                results[f'dbscan_{name}'] = dbscan.fit_predict(embedding)

                # Hierarchical
                agglo = AgglomerativeClustering(n_clusters=k)
                results[f'hierarchical_{name}'] = agglo.fit_predict(embedding)

            except Exception as e:
                print(f"Error in clustering for {name}: {str(e)}")
                continue

        return results

    # Generate visualization for each clustering method
    def analyze_individual_methods(self, data, output_dir):
        results = {}

        # Store embeddings for later use
        embeddings = self.multi_embedding(data)
        self.embeddings = embeddings

        # Clustering for each embedding
        for name, embedding in embeddings.items():
            k = min(10, data.shape[0] // self.min_cluster_size)

            # K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans_clusters = kmeans.fit_predict(embedding)
            self.plot_clustering(embedding, kmeans_clusters,
                                 f'K-means on {name.upper()}',
                                 os.path.join(output_dir, f'kmeans_{name}_clustering.png'))
            results[f'kmeans_{name}'] = kmeans_clusters

            # DBSCAN
            eps = np.mean(np.std(embedding, axis=0)) * 0.5
            dbscan = DBSCAN(eps=eps, min_samples=5, n_jobs=-1)
            dbscan_clusters = dbscan.fit_predict(embedding)
            self.plot_clustering(embedding, dbscan_clusters,
                                 f'DBSCAN on {name.upper()}',
                                 os.path.join(output_dir, f'dbscan_{name}_clustering.png'))
            results[f'dbscan_{name}'] = dbscan_clusters

            # Hierarchical
            if embedding.shape[0] < 100000:  # Only for smaller datasets
                hierarchical = AgglomerativeClustering(n_clusters=k)
                hierarchical_clusters = hierarchical.fit_predict(embedding)
                self.plot_clustering(embedding, hierarchical_clusters,
                                     f'Hierarchical on {name.upper()}',
                                     os.path.join(output_dir, f'hierarchical_{name}_clustering.png'))
                results[f'hierarchical_{name}'] = hierarchical_clusters

        # Save comparison
        self.save_clustering_statistics(results, output_dir)
        return results

    def plot_clustering(self, embedding, clusters, title, save_path):
        plt.figure(figsize=(12, 12))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                              c=clusters, cmap='tab20', s=5)
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def run_analysis(self, expression_matrix, output_dir):
        start_time = time.time()
        print("Starting complete analysis...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Preprocess data
        processed_data = self.preprocess_data(expression_matrix)

        # Run clustering analysis
        results = self.analyze_individual_methods(processed_data, output_dir)

        print(f"Analysis complete. Total time: {time.time() - start_time:.2f} seconds")
        return results

    def save_clustering_statistics(self, results, output_dir):
        """Save statistics for each clustering method"""
        stats = []
        for method, clusters in results.items():
            unique_clusters = np.unique(clusters[clusters >= 0])

            # Calculate silhouette score
            embedding_name = method.split('_')[-1]
            embedding = self.embeddings[embedding_name]

            valid_mask = clusters >= 0
            if len(unique_clusters) > 1 and np.sum(valid_mask) > 0:
                silhouette = silhouette_score(
                    embedding[valid_mask],
                    clusters[valid_mask]
                )
            else:
                silhouette = -1.0

            stat = {
                'Method': method,
                'Num_Clusters': len(unique_clusters),
                'Silhouette_Score': silhouette,
                'Noise_Points': np.sum(clusters < 0),
                'Largest_Cluster': np.max([np.sum(clusters == c) for c in unique_clusters]),
                'Smallest_Cluster': np.min([np.sum(clusters == c) for c in unique_clusters]),
                'Mean_Cluster_Size': np.mean([np.sum(clusters == c) for c in unique_clusters])
            }
            stats.append(stat)
            print(f"{method}: Silhouette Score = {silhouette:.3f}")

        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(os.path.join(output_dir, 'clustering_statistics.csv'), index=False)

        print("\n" + "=" * 60)
        print("SILHOUETTE SCORE SUMMARY")
        print("=" * 60)
        print(f"Average Silhouette Score: {stats_df['Silhouette_Score'].mean():.3f}")
        print(f"Best Method: {stats_df.loc[stats_df['Silhouette_Score'].idxmax(), 'Method']}")
        print(f"Best Score: {stats_df['Silhouette_Score'].max():.3f}")
        print("=" * 60)


def process_gef_file(input_file, output_dir):
    print("Reading GEF file...")
    with h5py.File(input_file, 'r') as f:
        cell_data = f['cellBin/cell'][:]
        cell_exp = f['cellBin/cellExp'][:]
        genes = f['cellBin/gene'][:]

        coordinates = np.empty((len(cell_data), 2), dtype=np.float32)
        coordinates[:, 0] = cell_data['x']
        coordinates[:, 1] = cell_data['y']

        print("Creating expression matrix...")
        rows = []
        cols = []
        data = []

        for i, cell in enumerate(cell_data):
            if i % 1000 == 0:
                print(f"Processing cell {i}/{len(cell_data)}")

            start = cell['offset']
            end = start + cell['geneCount']
            cell_genes = cell_exp[start:end]

            nonzero_mask = cell_genes['count'] > 0
            if np.any(nonzero_mask):
                rows.extend([i] * np.sum(nonzero_mask))
                cols.extend(cell_genes['geneID'][nonzero_mask])
                data.extend(cell_genes['count'][nonzero_mask])

        print("Converting to sparse matrix...")
        expression_matrix = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(len(cell_data), len(genes)),
            dtype=np.float32
        ).tocsr()

    print("Running analysis...")
    rec = REClustering(min_cluster_size=100, max_depth=3)
    results = rec.run_analysis(expression_matrix.toarray(), output_dir)

    return results, genes


if __name__ == "__main__":
    # Default output groups results under output/baseline/<run timestamp>/ so
    # repeated runs don't overwrite each other. All stages in one orchestrated
    # run (run_pipeline.sh / watch_runner.py) share the same timestamp via
    # $ST_RUN_ID; a standalone run generates its own.
    _run_id = os.environ.get("ST_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
    _default_output = os.environ.get("ST_OUTPUT_DIR", os.path.join("output", "baseline", _run_id))

    # Default input: $ST_INPUT_GEF, else auto-detect if exactly one .gef file
    # exists in the current directory. No hardcoded filename — this works with
    # anyone's data, not just one specific sample.
    _env_input = os.environ.get("ST_INPUT_GEF")
    _gef_candidates = sorted(glob.glob("*.gef"))
    _default_input = _env_input or (_gef_candidates[0] if len(_gef_candidates) == 1 else None)

    parser = argparse.ArgumentParser(
        description="Baseline recursive clustering on a Stereo-seq GEF file."
    )
    parser.add_argument(
        "--input", dest="input_file",
        default=_default_input,
        help="Path to the input .gef file. Defaults to $ST_INPUT_GEF, or the single "
             ".gef file in the current directory if there's exactly one.",
    )
    parser.add_argument(
        "--output", dest="output_dir",
        default=_default_output,
        help="Directory to write results to (default: output/baseline/<run timestamp>; "
             "can also be set via $ST_OUTPUT_DIR)",
    )
    args = parser.parse_args()
    if not args.input_file:
        n = len(_gef_candidates)
        if n == 0:
            parser.error("No --input given, $ST_INPUT_GEF not set, and no .gef file "
                          "found in the current directory. Specify --input path/to/file.gef.")
        else:
            parser.error(f"No --input given and {n} .gef files found in the current "
                          f"directory ({', '.join(_gef_candidates)}) — ambiguous. "
                          f"Specify --input path/to/file.gef.")
    input_file = args.input_file
    output_dir = args.output_dir

    start_time = time.time()
    results, genes = process_gef_file(input_file, output_dir)

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
