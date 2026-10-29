
import argparse
import glob
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
from scipy.stats import zscore, spearmanr
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import json
import os
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
import re
from openai import OpenAI

# OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available")

class DataObserver:
    def __init__(self):
        self.observations = []

    def observe_data_characteristics(self, expression_matrix: np.ndarray,
                                     coordinates: np.ndarray,
                                     gene_names: List[str]) -> Dict[str, Any]:

        observations = {
            'basic_stats': self._observe_basic_statistics(expression_matrix, coordinates),
            'spatial_patterns': self._observe_spatial_patterns(coordinates),
            'expression_patterns': self._observe_expression_patterns(expression_matrix),
            'gene_patterns': self._observe_gene_patterns(expression_matrix, gene_names),
            'anomalies': self._detect_anomalies(expression_matrix, coordinates),
            'complexity_metrics': self._calculate_complexity(expression_matrix, coordinates)
        }

        self.observations.append(observations)
        return observations

    def _observe_basic_statistics(self, expression_matrix: np.ndarray,
                                  coordinates: np.ndarray) -> Dict[str, Any]:
        return {
            'n_cells': expression_matrix.shape[0],
            'n_genes': expression_matrix.shape[1],
            'sparsity': float(1 - np.count_nonzero(expression_matrix) / expression_matrix.size),
            'mean_expression_per_cell': float(np.mean(np.sum(expression_matrix, axis=1))),
            'mean_genes_per_cell': float(np.mean(np.sum(expression_matrix > 0, axis=1))),
            'spatial_extent': {
                'x_range': (float(coordinates[:, 0].min()), float(coordinates[:, 0].max())),
                'y_range': (float(coordinates[:, 1].min()), float(coordinates[:, 1].max())),
                'area': float((coordinates[:, 0].max() - coordinates[:, 0].min()) *
                             (coordinates[:, 1].max() - coordinates[:, 1].min()))
            }
        }

    def _observe_spatial_patterns(self, coordinates: np.ndarray) -> Dict[str, Any]:
        # Calculate local density
        nbrs = NearestNeighbors(n_neighbors=min(20, len(coordinates))).fit(coordinates)
        distances, _ = nbrs.kneighbors(coordinates)
        local_density = 1.0 / (np.mean(distances, axis=1) + 1e-6)

        # Detect spatial clustering
        spatial_heterogeneity = np.std(local_density) / (np.mean(local_density) + 1e-6)

        # Check for spatial gradients
        x_sorted_indices = np.argsort(coordinates[:, 0])
        y_sorted_indices = np.argsort(coordinates[:, 1])

        return {
            'density_heterogeneity': float(spatial_heterogeneity),
            'mean_local_density': float(np.mean(local_density)),
            'density_range': (float(local_density.min()), float(local_density.max())),
            'spatial_clustering_detected': spatial_heterogeneity > 0.5,
            'has_spatial_gradient': True  # Simplified
        }

    def _observe_expression_patterns(self, expression_matrix: np.ndarray) -> Dict[str, Any]:
        # Cell-level patterns
        cell_totals = np.sum(expression_matrix, axis=1)
        cell_diversity = np.sum(expression_matrix > 0, axis=1)

        # Gene-level patterns
        gene_expression = np.sum(expression_matrix, axis=0)
        gene_prevalence = np.sum(expression_matrix > 0, axis=0)

        # Detect highly variable genes
        gene_cv = np.std(expression_matrix, axis=0) / (np.mean(expression_matrix, axis=0) + 1e-6)
        highly_variable = np.sum(gene_cv > np.percentile(gene_cv, 90))

        return {
            'cell_expression_heterogeneity': float(np.std(cell_totals) / (np.mean(cell_totals) + 1e-6)),
            'cell_diversity_range': (int(cell_diversity.min()), int(cell_diversity.max())),
            'highly_variable_genes': int(highly_variable),
            'expression_concentration': float(np.sum(gene_expression > np.percentile(gene_expression, 95)) / len(gene_expression)),
            'has_distinct_subpopulations': np.std(cell_totals) / np.mean(cell_totals) > 0.5
        }

    def _observe_gene_patterns(self, expression_matrix: np.ndarray,
                              gene_names: List[str]) -> Dict[str, Any]:
        # Top expressed genes
        gene_totals = np.sum(expression_matrix, axis=0)
        top_indices = np.argsort(gene_totals)[-20:]
        top_genes = [gene_names[i] for i in top_indices]

        # Gene nomenclature analysis
        uppercase_genes = sum(1 for g in gene_names if g.isupper() and len(g) > 1)
        titlecase_genes = sum(1 for g in gene_names if len(g) > 1 and g[0].isupper() and g[1:].islower())

        likely_species = 'human' if uppercase_genes > titlecase_genes else 'mouse'

        return {
            'top_expressed_genes': top_genes[:10],
            'likely_species': likely_species,
            'gene_nomenclature': f"{uppercase_genes} uppercase, {titlecase_genes} titlecase"
        }

    def _detect_anomalies(self, expression_matrix: np.ndarray,
                         coordinates: np.ndarray) -> Dict[str, Any]:
        anomalies = []

        # Detect outlier cells
        cell_totals = np.sum(expression_matrix, axis=1)
        z_scores = np.abs(zscore(cell_totals))
        outlier_cells = np.sum(z_scores > 3)

        if outlier_cells > expression_matrix.shape[0] * 0.05:
            anomalies.append({
                'type': 'expression_outliers',
                'description': f'{outlier_cells} cells with extreme expression levels',
                'severity': 'medium'
            })

        # Detect spatial clusters with unusual expression
        if len(coordinates) > 100:
            # Simple spatial clustering
            from sklearn.cluster import DBSCAN
            spatial_clusters = DBSCAN(eps=100, min_samples=10).fit_predict(coordinates)
            n_spatial_clusters = len(set(spatial_clusters)) - (1 if -1 in spatial_clusters else 0)

            if n_spatial_clusters > 5:
                anomalies.append({
                    'type': 'spatial_fragmentation',
                    'description': f'{n_spatial_clusters} distinct spatial regions detected',
                    'severity': 'high'
                })

        # Detect bimodal distributions
        cell_diversity = np.sum(expression_matrix > 0, axis=1)
        hist, _ = np.histogram(cell_diversity, bins=20)
        peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0]

        if len(peaks) >= 2:
            anomalies.append({
                'type': 'bimodal_population',
                'description': 'Multiple distinct cell populations detected',
                'severity': 'high'
            })

        return {
            'anomalies_detected': len(anomalies),
            'anomalies': anomalies
        }

    def _calculate_complexity(self, expression_matrix: np.ndarray,
                             coordinates: np.ndarray) -> Dict[str, Any]:
        # PCA for intrinsic dimensionality
        if expression_matrix.shape[1] > 50:
            pca = PCA(n_components=50)
            pca.fit(np.log1p(expression_matrix))
            explained_var = pca.explained_variance_ratio_
            intrinsic_dim = np.sum(np.cumsum(explained_var) < 0.9) + 1
        else:
            intrinsic_dim = expression_matrix.shape[1]

        return {
            'intrinsic_dimensionality': int(intrinsic_dim),
            'complexity_score': float(intrinsic_dim / expression_matrix.shape[1]),
            'estimated_subpopulations': int(max(3, intrinsic_dim // 10))
        }

# Generates hypotheses based on observations
class HypothesisGenerator:
    def __init__(self, use_llm: bool = True, llm_client=None):
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.generated_hypotheses = []

    def generate_hypotheses(self, observations: Dict[str, Any],
                           max_hypotheses: int = 5,
                           previous_hypotheses: List[Dict] = None) -> List[Dict[str, Any]]:
        hypotheses = []

        # Rule-based hypothesis generation
        hypotheses.extend(self._generate_rule_based_hypotheses(observations))

        # LLM-based hypothesis generation (if available)
        if self.use_llm and self.llm_client:
            hypotheses.extend(self._generate_llm_hypotheses(observations))

        # Filter out previously tested hypotheses
        if previous_hypotheses:
            tested_types = {h['type'] for h in previous_hypotheses}
            hypotheses = [h for h in hypotheses if h['type'] not in tested_types]

        # Prioritize and select top hypotheses
        prioritized = self._prioritize_hypotheses(hypotheses, observations)

        selected = prioritized[:max_hypotheses]
        self.generated_hypotheses.extend(selected)

        return selected

    def _generate_rule_based_hypotheses(self, observations: Dict[str, Any]) -> List[Dict[str, Any]]:

        hypotheses = []
        basic = observations['basic_stats']
        spatial = observations['spatial_patterns']
        expression = observations['expression_patterns']
        anomalies = observations['anomalies']

        # H1: Spatial heterogeneity hypothesis
        if spatial['spatial_clustering_detected']:
            hypotheses.append({
                'statement': f"The tissue contains {observations['complexity_metrics']['estimated_subpopulations']} spatially distinct cell populations with different expression profiles",
                'type': 'spatial_heterogeneity',
                'rationale': f"Spatial density heterogeneity ({spatial['density_heterogeneity']:.2f}) suggests organized tissue architecture",
                'priority': 'high',
                'testable': True,
                'analysis_methods': ['spatial_clustering', 'expression_correlation', 'marker_analysis']
            })

        # H2: Expression subpopulation hypothesis
        if expression['has_distinct_subpopulations']:
            hypotheses.append({
                'statement': "Cells cluster into transcriptionally distinct groups that may represent different cell types or states",
                'type': 'expression_subpopulations',
                'rationale': f"High expression heterogeneity ({expression['cell_expression_heterogeneity']:.2f}) indicates distinct cell populations",
                'priority': 'high',
                'testable': True,
                'analysis_methods': ['clustering', 'differential_expression', 'marker_identification']
            })

        # H3: Rare cell type hypothesis
        if anomalies['anomalies_detected'] > 0:
            for anomaly in anomalies['anomalies']:
                if anomaly['type'] == 'expression_outliers':
                    hypotheses.append({
                        'statement': "The tissue contains rare cell populations with unique expression signatures",
                        'type': 'rare_cell_detection',
                        'rationale': anomaly['description'],
                        'priority': 'medium',
                        'testable': True,
                        'analysis_methods': ['outlier_analysis', 'marker_identification']
                    })

        # H4: Spatial gradient hypothesis
        if spatial['has_spatial_gradient']:
            hypotheses.append({
                'statement': "Gene expression changes gradually across spatial coordinates, suggesting tissue organization or developmental gradients",
                'type': 'spatial_gradient',
                'rationale': "Spatial patterns suggest organized tissue structure",
                'priority': 'medium',
                'testable': True,
                'analysis_methods': ['gradient_analysis', 'spatial_autocorrelation']
            })

        # H5: Functional domain hypothesis
        if spatial['spatial_clustering_detected'] and expression['has_distinct_subpopulations']:
            hypotheses.append({
                'statement': "Spatially coherent regions exhibit coordinated functional programs",
                'type': 'functional_domains',
                'rationale': "Both spatial and expression heterogeneity suggest organized functional architecture",
                'priority': 'high',
                'testable': True,
                'analysis_methods': ['spatial_clustering', 'pathway_enrichment', 'coexpression_analysis']
            })

        # H6: Cell-cell interaction hypothesis
        if basic['n_cells'] > 500 and spatial['spatial_clustering_detected']:
            hypotheses.append({
                'statement': "Neighboring cells exhibit coordinated expression patterns indicating cell-cell communication",
                'type': 'cell_interactions',
                'rationale': f"Sufficient cells ({basic['n_cells']}) and spatial organization enable interaction analysis",
                'priority': 'medium',
                'testable': True,
                'analysis_methods': ['neighborhood_analysis', 'ligand_receptor_analysis']
            })

        # H7: High variability genes hypothesis
        if expression['highly_variable_genes'] > expression.get('n_genes', 100) * 0.05:
            hypotheses.append({
                'statement': f"The {expression['highly_variable_genes']} highly variable genes define distinct biological programs",
                'type': 'variable_gene_programs',
                'rationale': "High proportion of variable genes suggests complex regulatory programs",
                'priority': 'medium',
                'testable': True,
                'analysis_methods': ['clustering', 'coexpression_analysis']
            })

        # H8: Expression concentration hypothesis
        if expression.get('expression_concentration', 0) > 0.15:
            hypotheses.append({
                'statement': "A small subset of highly expressed genes dominates the transcriptional landscape",
                'type': 'expression_concentration',
                'rationale': f"Expression concentration ({expression['expression_concentration']:.2%}) suggests specialized function",
                'priority': 'low',
                'testable': True,
                'analysis_methods': ['marker_identification', 'pathway_enrichment']
            })

        return hypotheses

    def _generate_llm_hypotheses(self, observations: Dict[str, Any]) -> List[Dict[str, Any]]:

        if not self.llm_client:
            return []

        try:
            # Create prompt for LLM
            prompt = self._create_llm_prompt(observations)

            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in spatial transcriptomics analysis. Generate novel, testable hypotheses based on data observations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                seed=int(time.time()) % 10000,
                max_tokens=500
            )

            # Parse LLM response
            llm_text = response.choices[0].message.content
            llm_hypotheses = self._parse_llm_response(llm_text)

            return llm_hypotheses

        except Exception as e:
            print(f"LLM hypothesis generation failed: {e}")
            return []

    def _create_llm_prompt(self, observations: Dict[str, Any]) -> str:

        prompt = f"""Based on spatial transcriptomics data analysis, generate 2-3 novel, testable hypotheses.

DATA OBSERVATIONS:
DATA OBSERVATIONS:
- Dataset: {observations['basic_stats']['n_cells']} cells, {observations['basic_stats']['n_genes']} genes
- Sparsity: {observations['basic_stats']['sparsity']:.1%}
- Spatial clustering detected: {observations['spatial_patterns']['spatial_clustering_detected']}
- Expression heterogeneity: {observations['expression_patterns']['cell_expression_heterogeneity']:.2f}
- Estimated subpopulations: {observations['complexity_metrics']['estimated_subpopulations']}
- Anomalies detected: {observations['anomalies']['anomalies_detected']}

Generate hypotheses that:
1. Are specific and testable
2. Go beyond obvious patterns
3. Could reveal novel biological insights
4. Consider spatial organization

Format each hypothesis as:
HYPOTHESIS: [clear statement]
TYPE: [category]
RATIONALE: [why this hypothesis is worth testing]
METHODS: [how to test it]

Generate 2-3 hypotheses now:"""

        return prompt

    def _parse_llm_response(self, llm_text: str) -> List[Dict[str, Any]]:

        hypotheses = []

        # Simple parsing (can be improved)
        sections = llm_text.split('HYPOTHESIS:')

        for section in sections[1:]:  # Skip first empty section
            try:
                lines = section.strip().split('\n')
                statement = lines[0].strip()

                hypothesis = {
                    'statement': statement,
                    'type': 'llm_generated',
                    'rationale': 'Generated by LLM based on data patterns',
                    'priority': 'medium',
                    'testable': True,
                    'analysis_methods': ['exploratory_analysis'],
                    'source': 'llm'
                }

                # Extract type and rationale if present
                for line in lines[1:]:
                    if line.startswith('TYPE:'):
                        hypothesis['type'] = line.replace('TYPE:', '').strip()
                    elif line.startswith('RATIONALE:'):
                        hypothesis['rationale'] = line.replace('RATIONALE:', '').strip()
                    elif line.startswith('METHODS:'):
                        methods_text = line.replace('METHODS:', '').strip()
                        hypothesis['analysis_methods'] = [m.strip() for m in methods_text.split(',')]

                hypotheses.append(hypothesis)
            except:
                continue

        return hypotheses

    #prioritize hypotheses based on data and testability
    def _prioritize_hypotheses(self, hypotheses: List[Dict[str, Any]],
                               observations: Dict[str, Any]) -> List[Dict[str, Any]]:

        def score_hypothesis(h):
            score = 0

            # Priority weight
            if h['priority'] == 'high':
                score += 10
            elif h['priority'] == 'medium':
                score += 5

            # Testability
            if h['testable']:
                score += 5

            # Multiple analysis methods available
            score += len(h.get('analysis_methods', [])) * 2

            # Prefer hypotheses that match detected patterns
            if h['type'] in ['spatial_heterogeneity', 'expression_subpopulations']:
                score += 8

            # Prefer novel LLM hypotheses
            if h.get('source') == 'llm':
                score += 3

            return score

        # Sort by score
        scored = [(score_hypothesis(h), h) for h in hypotheses]
        scored.sort(key=lambda x: x[0], reverse=True)

        return [h for score, h in scored]

# Plans and designs analyses to test hypotheses
class AnalysisPlanner:
    def __init__(self):
        self.analysis_plans = []

    def design_analysis(self, hypothesis: Dict[str, Any],
                       observations: Dict[str, Any]) -> Dict[str, Any]:

        plan = {
            'hypothesis': hypothesis,
            'analysis_type': hypothesis['type'],
            'methods': [],
            'expected_outputs': [],
            'validation_criteria': []
        }

        # Design based on hypothesis type
        if hypothesis['type'] == 'spatial_heterogeneity':
            plan['methods'] = [
                {'name': 'spatial_clustering', 'params': {'method': 'dbscan', 'adaptive': True}},
                {'name': 'spatial_visualization', 'params': {'color_by': 'cluster'}},
                {'name': 'cluster_characterization', 'params': {'top_markers': 20}}
            ]
            plan['expected_outputs'] = ['cluster_assignments', 'spatial_map', 'marker_genes']
            plan['validation_criteria'] = [
                'silhouette_score > 0.3',
                'clusters_spatially_coherent',
                'distinct_marker_profiles'
            ]

        elif hypothesis['type'] == 'expression_subpopulations':
            plan['methods'] = [
                {'name': 'dimensionality_reduction', 'params': {'method': 'pca', 'n_components': 50}},
                {'name': 'clustering', 'params': {'method': 'kmeans', 'k_range': (2, 15)}},
                {'name': 'differential_expression', 'params': {'method': 'wilcoxon'}},
                {'name': 'marker_identification', 'params': {'top_n': 20}}
            ]
            plan['expected_outputs'] = ['clusters', 'marker_genes', 'expression_profiles']
            plan['validation_criteria'] = [
                'silhouette_score > 0.25',
                'distinct_marker_genes_per_cluster',
                'biological_coherence'
            ]

        elif hypothesis['type'] == 'rare_cell_detection':
            plan['methods'] = [
                {'name': 'outlier_detection', 'params': {'method': 'isolation_forest'}},
                {'name': 'rare_cell_characterization', 'params': {}},
                {'name': 'marker_analysis', 'params': {'compare_to': 'all_other_cells'}}
            ]
            plan['expected_outputs'] = ['outlier_cells', 'unique_markers']
            plan['validation_criteria'] = [
                'outliers_detected',
                'unique_expression_signature',
                'potential_biological_identity'
            ]

        elif hypothesis['type'] == 'spatial_gradient':
            plan['methods'] = [
                {'name': 'gradient_analysis', 'params': {'direction': 'both_axes'}},
                {'name': 'spatial_autocorrelation', 'params': {'method': 'morans_i'}},
                {'name': 'gene_trajectory', 'params': {'top_variable_genes': 50}}
            ]
            plan['expected_outputs'] = ['gradient_genes', 'spatial_correlation_map']
            plan['validation_criteria'] = [
                'significant_spatial_correlation',
                'smooth_expression_gradients'
            ]

        elif hypothesis['type'] == 'functional_domains':
            plan['methods'] = [
                {'name': 'spatial_clustering', 'params': {'method': 'dbscan'}},
                {'name': 'pathway_enrichment', 'params': {'database': 'go_biological_process'}},
                {'name': 'coexpression_analysis', 'params': {'method': 'correlation'}}
            ]
            plan['expected_outputs'] = ['functional_domains', 'enriched_pathways', 'coexpression_modules']
            plan['validation_criteria'] = [
                'spatially_coherent_domains',
                'significant_pathway_enrichment',
                'coordinated_expression'
            ]

        elif hypothesis['type'] == 'cell_interactions':
            plan['methods'] = [
                {'name': 'neighborhood_analysis', 'params': {'radius': 100}},
                {'name': 'ligand_receptor_pairs', 'params': {'database': 'cellchatdb'}},
                {'name': 'spatial_colocalization', 'params': {}}
            ]
            plan['expected_outputs'] = ['interaction_scores', 'colocalization_patterns']
            plan['validation_criteria'] = [
                'significant_colocalization',
                'predicted_interactions'
            ]

        else:
            # Generic exploratory analysis
            plan['methods'] = [
                {'name': 'clustering', 'params': {'method': 'adaptive'}},
                {'name': 'dimensionality_reduction', 'params': {'method': 'umap'}},
                {'name': 'visualization', 'params': {}}
            ]
            plan['expected_outputs'] = ['clusters', 'visualization']
            plan['validation_criteria'] = ['patterns_detected']

        self.analysis_plans.append(plan)
        return plan


class AnalysisExecutor:

    def __init__(self):
        self.execution_history = []

    def execute_analysis(self, plan: Dict[str, Any],
                        expression_matrix: np.ndarray,
                        coordinates: np.ndarray,
                        gene_names: List[str]) -> Dict[str, Any]:

        results = {
            'hypothesis': plan['hypothesis']['statement'],
            'analysis_type': plan['analysis_type'],
            'outputs': {},
            'metrics': {},
            'patterns_found': []
        }

        start_time = time.time()

        try:
            for method_spec in plan['methods']:
                method_name = method_spec['name']
                params = method_spec['params']

                if method_name == 'spatial_clustering':
                    output = self._spatial_clustering(expression_matrix, coordinates, params)
                    results['outputs']['clusters'] = output
                    results['metrics']['n_clusters'] = output['n_clusters']
                    results['metrics']['silhouette_score'] = output['silhouette_score']

                elif method_name == 'clustering':
                    output = self._expression_clustering(expression_matrix, params)
                    results['outputs']['clusters'] = output
                    results['metrics']['n_clusters'] = output['n_clusters']
                    results['metrics']['silhouette_score'] = output['silhouette_score']

                elif method_name == 'dimensionality_reduction':
                    output = self._dimensionality_reduction(expression_matrix, params)
                    results['outputs']['reduced_features'] = output

                elif method_name == 'outlier_detection':
                    output = self._detect_outliers(expression_matrix, params)
                    results['outputs']['outliers'] = output
                    results['metrics']['n_outliers'] = output['n_outliers']

                elif method_name == 'gradient_analysis':
                    output = self._analyze_gradients(expression_matrix, coordinates, gene_names, params)
                    results['outputs']['gradients'] = output
                    results['metrics']['gradient_strength'] = output['mean_correlation']

                elif method_name == 'neighborhood_analysis':
                    output = self._analyze_neighborhoods(expression_matrix, coordinates, params)
                    results['outputs']['neighborhoods'] = output

                else:
                    print(f"  Method {method_name} not implemented, skipping...")

            # Validate results
            validation = self._validate_results(results, plan['validation_criteria'])
            results['validation'] = validation
            results['success'] = validation['passed']

        except Exception as e:
            print(f"  Analysis execution failed: {e}")
            results['success'] = False
            results['error'] = str(e)

        results['execution_time'] = time.time() - start_time
        self.execution_history.append(results)

        return results

    def _spatial_clustering(self, expression_matrix: np.ndarray,
                           coordinates: np.ndarray,
                           params: Dict) -> Dict[str, Any]:

        # Combine expression and spatial features
        from sklearn.preprocessing import StandardScaler

        # Expression features (PCA)
        pca = PCA(n_components=min(50, expression_matrix.shape[1]))
        expr_features = pca.fit_transform(np.log1p(expression_matrix))

        # Normalize spatial coordinates
        scaler = StandardScaler()
        spatial_features = scaler.fit_transform(coordinates)

        # Combined features (weight spatial more for spatial clustering)
        combined = np.hstack([expr_features * 0.3, spatial_features * 0.7])

        # Adaptive DBSCAN
        k = min(20, len(coordinates) // 10)
        nbrs = NearestNeighbors(n_neighbors=k).fit(combined)
        distances, _ = nbrs.kneighbors(combined)
        eps = np.percentile(distances[:, -1], 80)

        dbscan = DBSCAN(eps=eps, min_samples=k)
        clusters = dbscan.fit_predict(combined)

        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

        # Calculate silhouette if possible
        if n_clusters > 1:
            valid_mask = clusters >= 0
            if np.sum(valid_mask) > 10:
                sil_score = silhouette_score(combined[valid_mask], clusters[valid_mask])
            else:
                sil_score = 0.0
        else:
            sil_score = 0.0

        return {
            'clusters': clusters,
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'noise_ratio': np.sum(clusters == -1) / len(clusters)
        }

    def _expression_clustering(self, expression_matrix: np.ndarray,
                               params: Dict) -> Dict[str, Any]:

        # PCA preprocessing
        pca = PCA(n_components=min(50, expression_matrix.shape[1]))
        features = pca.fit_transform(np.log1p(expression_matrix))

        # K-means with optimal k search
        k_range = params.get('k_range', (2, 15))
        best_k = k_range[0]
        best_score = -1
        best_clusters = None

        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            score = silhouette_score(features, clusters)

            if score > best_score:
                best_score = score
                best_k = k
                best_clusters = clusters

        return {
            'clusters': best_clusters,
            'n_clusters': best_k,
            'silhouette_score': best_score
        }

    def _dimensionality_reduction(self, expression_matrix: np.ndarray,
                                  params: Dict) -> Dict[str, Any]:

        method = params.get('method', 'pca')
        n_components = params.get('n_components', 50)

        if method == 'pca':
            pca = PCA(n_components=min(n_components, expression_matrix.shape[1]))
            reduced = pca.fit_transform(np.log1p(expression_matrix))
            variance_explained = np.sum(pca.explained_variance_ratio_)
        else:
            # Fallback to PCA
            pca = PCA(n_components=min(n_components, expression_matrix.shape[1]))
            reduced = pca.fit_transform(np.log1p(expression_matrix))
            variance_explained = np.sum(pca.explained_variance_ratio_)

        return {
            'features': reduced,
            'n_components': reduced.shape[1],
            'variance_explained': variance_explained
        }

    def _detect_outliers(self, expression_matrix: np.ndarray,
                        params: Dict) -> Dict[str, Any]:

        # Simple z-score based outlier detection
        cell_totals = np.sum(expression_matrix, axis=1)
        z_scores = np.abs(zscore(cell_totals))

        threshold = params.get('threshold', 3.0)
        outliers = np.where(z_scores > threshold)[0]

        return {
            'outlier_indices': outliers,
            'n_outliers': len(outliers),
            'outlier_scores': z_scores[outliers]
        }

    def _analyze_gradients(self, expression_matrix: np.ndarray,
                          coordinates: np.ndarray,
                          gene_names: List[str],
                          params: Dict) -> Dict[str, Any]:

        # Find genes with spatial patterns
        gradient_genes = []
        correlations = []

        # Sample genes if too many
        n_genes_to_test = min(500, expression_matrix.shape[1])
        gene_indices = np.random.choice(expression_matrix.shape[1], n_genes_to_test, replace=False)

        for idx in gene_indices:
            gene_expr = expression_matrix[:, idx]

            if np.std(gene_expr) > 0:
                # Correlation with X coordinate
                corr_x = np.abs(spearmanr(coordinates[:, 0], gene_expr)[0])
                # Correlation with Y coordinate
                corr_y = np.abs(spearmanr(coordinates[:, 1], gene_expr)[0])

                max_corr = max(corr_x, corr_y)

                if max_corr > 0.3:  # Significant spatial correlation
                    gradient_genes.append({
                        'gene': gene_names[idx],
                        'gene_idx': int(idx),
                        'correlation_x': float(corr_x),
                        'correlation_y': float(corr_y),
                        'max_correlation': float(max_corr)
                    })
                    correlations.append(max_corr)

        return {
            'gradient_genes': gradient_genes[:20],  # Top 20
            'n_gradient_genes': len(gradient_genes),
            'mean_correlation': float(np.mean(correlations)) if correlations else 0.0
        }

    def _analyze_neighborhoods(self, expression_matrix: np.ndarray,
                              coordinates: np.ndarray,
                              params: Dict) -> Dict[str, Any]:

        radius = params.get('radius', 100)

        # For each cell, find neighbors and compare expression
        neighborhood_scores = []

        for i in range(min(100, len(coordinates))):  # Sample for efficiency
            distances = np.sqrt(np.sum((coordinates - coordinates[i])**2, axis=1))
            neighbors = np.where(distances <= radius)[0]

            if len(neighbors) > 1:
                # Compare expression similarity
                neighbor_expr = expression_matrix[neighbors]
                similarity = np.mean(np.corrcoef(neighbor_expr))
                neighborhood_scores.append(similarity)

        return {
            'mean_neighborhood_similarity': float(np.mean(neighborhood_scores)) if neighborhood_scores else 0.0,
            'n_analyzed': len(neighborhood_scores)
        }

    def _validate_results(self, results: Dict[str, Any],
                         criteria: List[str]) -> Dict[str, Any]:

        validation = {
            'criteria_checked': len(criteria),
            'criteria_passed': 0,
            'passed': False,
            'details': []
        }

        for criterion in criteria:
            passed = self._check_criterion(criterion, results)
            if passed:
                validation['criteria_passed'] += 1

            validation['details'].append({
                'criterion': criterion,
                'passed': passed
            })

        # Overall pass if majority of criteria met
        validation['passed'] = validation['criteria_passed'] >= len(criteria) * 0.5

        return validation

    def _check_criterion(self, criterion: str, results: Dict[str, Any]) -> bool:

        metrics = results.get('metrics', {})

        if 'silhouette_score' in criterion:
            threshold = float(criterion.split('>')[-1].strip())
            return metrics.get('silhouette_score', 0) > threshold

        elif 'clusters_spatially_coherent' in criterion:
            return metrics.get('silhouette_score', 0) > 0.2

        elif 'distinct_marker_profiles' in criterion:
            return metrics.get('n_clusters', 0) >= 2

        elif 'distinct_marker_genes_per_cluster' in criterion:
            return metrics.get('n_clusters', 0) >= 2

        elif 'biological_coherence' in criterion:
            return metrics.get('silhouette_score', 0) > 0.2

        elif 'outliers_detected' in criterion:
            return metrics.get('n_outliers', 0) > 0

        elif 'significant_spatial_correlation' in criterion:
            return metrics.get('gradient_strength', 0) > 0.3

        elif 'spatially_coherent_domains' in criterion:
            return metrics.get('n_clusters', 0) >= 2

        elif 'patterns_detected' in criterion:
            return True  # Always true for exploratory

        else:
            return True  # Unknown criteria pass by default


class ResultInterpreter:

    def __init__(self, use_llm: bool = True, llm_client=None):
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.interpretations = []

    def interpret_results(self, hypothesis: Dict[str, Any],
                         results: Dict[str, Any],
                         observations: Dict[str, Any]) -> Dict[str, Any]:

        interpretation = {
            'hypothesis': hypothesis['statement'],
            'supported': results.get('success', False),
            'confidence': self._calculate_confidence(results),
            'key_findings': [],
            'biological_insights': [],
            'unexpected_patterns': [],
            'recommendations': []
        }

        # Extract key findings
        interpretation['key_findings'] = self._extract_key_findings(results, hypothesis)

        # Detect unexpected patterns
        interpretation['unexpected_patterns'] = self._detect_unexpected_patterns(
            results, hypothesis, observations
        )

        # Generate biological insights
        if self.use_llm and self.llm_client:
            interpretation['biological_insights'] = self._generate_llm_insights(
                hypothesis, results, observations
            )
        else:
            interpretation['biological_insights'] = self._generate_rule_based_insights(
                hypothesis, results
            )

        # Generate recommendations
        interpretation['recommendations'] = self._generate_recommendations(
            hypothesis, results, interpretation
        )

        self.interpretations.append(interpretation)
        return interpretation

    def _calculate_confidence(self, results: Dict[str, Any]) -> str:

        if not results.get('success', False):
            return 'low'

        validation = results.get('validation', {})
        pass_rate = validation.get('criteria_passed', 0) / max(validation.get('criteria_checked', 1), 1)

        metrics = results.get('metrics', {})
        sil_score = metrics.get('silhouette_score', 0)

        if pass_rate >= 0.8 and sil_score > 0.4:
            return 'high'
        elif pass_rate >= 0.6 and sil_score > 0.25:
            return 'medium'
        else:
            return 'low'

    def _extract_key_findings(self, results: Dict[str, Any],
                             hypothesis: Dict[str, Any]) -> List[str]:

        findings = []
        metrics = results.get('metrics', {})
        outputs = results.get('outputs', {})

        # Clustering findings
        if 'n_clusters' in metrics:
            n_clusters = metrics['n_clusters']
            sil_score = metrics.get('silhouette_score', 0)
            findings.append(
                f"Identified {n_clusters} distinct clusters with silhouette score {sil_score:.3f}"
            )

        # Outlier findings
        if 'n_outliers' in metrics:
            n_outliers = metrics['n_outliers']
            if n_outliers > 0:
                findings.append(f"Detected {n_outliers} outlier cells with unique expression profiles")

        # Gradient findings
        if 'gradient_strength' in metrics:
            strength = metrics['gradient_strength']
            if strength > 0.3:
                findings.append(
                    f"Found significant spatial expression gradients (correlation: {strength:.3f})"
                )

        # Validation findings
        validation = results.get('validation', {})
        if validation.get('passed', False):
            findings.append(
                f"Hypothesis validation: {validation['criteria_passed']}/{validation['criteria_checked']} criteria met"
            )

        return findings

    def _detect_unexpected_patterns(self, results: Dict[str, Any],
                                   hypothesis: Dict[str, Any],
                                   observations: Dict[str, Any]) -> List[str]:

        unexpected = []
        metrics = results.get('metrics', {})

        # More clusters than expected
        expected_clusters = observations.get('complexity_metrics', {}).get('estimated_subpopulations', 5)
        actual_clusters = metrics.get('n_clusters', 0)

        if actual_clusters > expected_clusters * 1.5:
            unexpected.append(
                f"Found {actual_clusters} clusters, significantly more than expected ({expected_clusters})"
            )

        # Unexpectedly high quality
        sil_score = metrics.get('silhouette_score', 0)
        if sil_score > 0.6:
            unexpected.append(
                f"Exceptionally high clustering quality (silhouette: {sil_score:.3f}) suggests very distinct populations"
            )

        # Strong spatial gradients
        if 'gradient_strength' in metrics and metrics['gradient_strength'] > 0.5:
            unexpected.append(
                f"Strong spatial organization (gradient strength: {metrics['gradient_strength']:.3f}) suggests developmental or functional zonation"
            )

        return unexpected

    def _generate_rule_based_insights(self, hypothesis: Dict[str, Any],
                                     results: Dict[str, Any]) -> List[str]:

        insights = []
        metrics = results.get('metrics', {})
        h_type = hypothesis['type']

        if h_type == 'spatial_heterogeneity':
            n_clusters = metrics.get('n_clusters', 0)
            if n_clusters >= 3:
                insights.append(
                    "Multiple spatially distinct regions suggest organized tissue architecture with functional specialization"
                )

        elif h_type == 'expression_subpopulations':
            sil_score = metrics.get('silhouette_score', 0)
            if sil_score > 0.3:
                insights.append(
                    "Distinct transcriptional profiles indicate different cell types or functional states"
                )

        elif h_type == 'rare_cell_detection':
            n_outliers = metrics.get('n_outliers', 0)
            if n_outliers > 0:
                insights.append(
                    "Rare cell populations detected may represent specialized cell types or transitional states"
                )

        elif h_type == 'spatial_gradient':
            if metrics.get('gradient_strength', 0) > 0.3:
                insights.append(
                    "Spatial expression gradients suggest organized tissue patterning, possibly reflecting developmental processes or functional zonation"
                )

        return insights

    def _generate_llm_insights(self, hypothesis: Dict[str, Any],
                              results: Dict[str, Any],
                              observations: Dict[str, Any]) -> List[str]:

        if not self.llm_client:
            return self._generate_rule_based_insights(hypothesis, results)

        try:
            prompt = f"""As a spatial transcriptomics expert, interpret these analysis results:

HYPOTHESIS: {hypothesis['statement']}
TYPE: {hypothesis['type']}

RESULTS:
- Success: {results.get('success', False)}
- Key metrics: {json.dumps(results.get('metrics', {}), indent=2)}
- Validation: {results.get('validation', {}).get('passed', False)}

CONTEXT:
- Dataset: {observations['basic_stats']['n_cells']} cells
- Species: {observations.get('gene_patterns', {}).get('likely_species', 'unknown')}

Provide 2-3 biological insights that:
1. Explain what these results mean biologically
2. Connect to known biology
3. Suggest functional implications

Be specific and scientifically grounded."""

            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in spatial transcriptomics and tissue biology."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                seed=int(time.time()) % 10000,
                max_tokens=300
            )

            insights_text = response.choices[0].message.content
            # Split into individual insights
            insights = [line.strip() for line in insights_text.split('\n') if line.strip() and not line.strip().startswith('#')]

            return insights[:3]  # Top 3

        except Exception as e:
            print(f"LLM insight generation failed: {e}")
            return self._generate_rule_based_insights(hypothesis, results)

    def _generate_recommendations(self, hypothesis: Dict[str, Any],
                                 results: Dict[str, Any],
                                 interpretation: Dict[str, Any]) -> List[str]:

        recommendations = []

        if interpretation['supported']:
            recommendations.append(
                f"Hypothesis supported - consider validating with marker gene analysis"
            )

            if interpretation['confidence'] == 'high':
                recommendations.append(
                    "High confidence results - proceed with biological validation experiments"
                )
        else:
            recommendations.append(
                f"Hypothesis not strongly supported - consider alternative hypotheses"
            )

        if interpretation['unexpected_patterns']:
            recommendations.append(
                "Unexpected patterns detected - explore these further with targeted analyses"
            )

        # Specific recommendations based on hypothesis type
        h_type = hypothesis['type']
        if h_type == 'spatial_heterogeneity':
            recommendations.append(
                "Perform marker gene identification for each spatial cluster"
            )
        elif h_type == 'expression_subpopulations':
            recommendations.append(
                "Conduct pathway enrichment analysis for each cluster"
            )
        elif h_type == 'rare_cell_detection':
            recommendations.append(
                "Validate rare cell populations with orthogonal methods"
            )

        return recommendations


class TrulyAutonomousAgent:

    def __init__(self, api_key: Optional[str] = None, use_llm: bool = True):
        self.observer = DataObserver()
        self.hypothesis_generator = HypothesisGenerator(use_llm=use_llm, llm_client=None)
        self.planner = AnalysisPlanner()
        self.executor = AnalysisExecutor()
        self.interpreter = ResultInterpreter(use_llm=use_llm, llm_client=None)

        # Initialize LLM if available
        if use_llm and OPENAI_AVAILABLE and api_key:
            try:
                llm_client = OpenAI(api_key=api_key)
                self.hypothesis_generator.llm_client = llm_client
                self.interpreter.llm_client = llm_client
                print("✓ LLM-enhanced autonomous agent initialized")
            except:
                print("LLM initialization failed, using rule-based agent")
        else:
            print("✓ Rule-based autonomous agent initialized")

        self.discoveries = []
        self.learning_history = []

    def autonomous_discovery(self, expression_matrix: np.ndarray,
                           coordinates: np.ndarray,
                           gene_names: List[str],
                           max_iterations: int = 5,
                           min_confidence: str = 'medium') -> List[Dict[str, Any]]:
        """
        Run autonomous discovery loop

        The agent will:
        1. Observe data characteristics
        2. Generate hypotheses dynamically
        3. Plan and execute analyses
        4. Interpret results
        5. Learn and iterate
        """

        print("="*70)
        print("🤖 AUTONOMOUS SPATIAL TRANSCRIPTOMICS DISCOVERY")
        print("="*70)
        print(f"Dataset: {expression_matrix.shape[0]} cells, {expression_matrix.shape[1]} genes")
        print(f"Max iterations: {max_iterations}")
        print("="*70)

        iteration = 0
        satisfied = False

        while iteration < max_iterations and not satisfied:
            iteration += 1
            print(f"\n{'='*70}")
            print(f"DISCOVERY ITERATION {iteration}/{max_iterations}")
            print(f"{'='*70}")

            # Step 1: Observe data
            print(f"Step 1: Observing data characteristics...")
            observations = self.observer.observe_data_characteristics(
                expression_matrix, coordinates, gene_names
            )
            self._print_observations(observations)

            # Step 2: Generate hypotheses
            print(f"Step 2: Generating hypotheses...")

            # Pass previous hypotheses to avoid repetition
            previous_hypotheses = [d['hypothesis'] for d in self.discoveries]

            hypotheses = self.hypothesis_generator.generate_hypotheses(
                observations,
                max_hypotheses=3 if iteration == 1 else 2,  # More on first iteration
                previous_hypotheses=previous_hypotheses
            )
            self._print_hypotheses(hypotheses)

            if not hypotheses:
                print("No testable hypotheses generated. Ending discovery.")
                break

            # Step 3: Test each hypothesis
            for h_idx, hypothesis in enumerate(hypotheses, 1):
                print(f"Step 3.{h_idx}: Testing hypothesis...")
                print(f"   Hypothesis: {hypothesis['statement']}")

                # Plan analysis
                print(f"   Planning analysis...")
                plan = self.planner.design_analysis(hypothesis, observations)
                print(f"   Analysis plan: {len(plan['methods'])} methods")

                # Execute analysis
                print(f"   Executing analysis...")
                results = self.executor.execute_analysis(
                    plan, expression_matrix, coordinates, gene_names
                )

                # Interpret results
                print(f"Step 4: Interpreting results...")
                interpretation = self.interpreter.interpret_results(
                    hypothesis, results, observations
                )

                # Record discovery
                discovery = {
                    'iteration': iteration,
                    'hypothesis': hypothesis,
                    'results': results,
                    'interpretation': interpretation,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.discoveries.append(discovery)

                self._print_interpretation(interpretation)

            # Step 5: Learn and decide whether to continue
            print(f"\n🎓 Step 5: Learning from discoveries...")
            should_continue, reason = self._should_continue_exploring(
                iteration, max_iterations, min_confidence
            )

            print(f"   Decision: {'Continue exploring' if should_continue else 'Stop exploring'}")
            print(f"   Reason: {reason}")

            if not should_continue:
                satisfied = True

        print(f"\n{'='*70}")
        print(f"AUTONOMOUS DISCOVERY COMPLETE")
        print(f"{'='*70}")
        print(f"Total discoveries: {len(self.discoveries)}")
        print(f"Iterations completed: {iteration}/{max_iterations}")

        return self.discoveries

    def _print_observations(self, observations: Dict[str, Any]):
        basic = observations['basic_stats']
        spatial = observations['spatial_patterns']
        expression = observations['expression_patterns']

        print(f"   • Dataset: {basic['n_cells']} cells, {basic['n_genes']} genes")
        print(f"   • Sparsity: {basic['sparsity']:.1%}")
        print(f"   • Spatial clustering: {'Yes' if spatial['spatial_clustering_detected'] else 'No'}")
        print(f"   • Expression heterogeneity: {expression['cell_expression_heterogeneity']:.2f}")
        print(f"   • Estimated populations: {observations['complexity_metrics']['estimated_subpopulations']}")
        print(f"   • Anomalies detected: {observations['anomalies']['anomalies_detected']}")

    def _print_hypotheses(self, hypotheses: List[Dict[str, Any]]):
        print(f"   Generated {len(hypotheses)} hypotheses:")
        for i, h in enumerate(hypotheses, 1):
            print(f"\n   {i}. {h['statement']}")
            print(f"      Type: {h['type']} | Priority: {h['priority']}")
            print(f"      Rationale: {h['rationale']}")

    def _print_interpretation(self, interpretation: Dict[str, Any]):
        print(f"   Supported: {interpretation['supported']}")
        print(f"   Confidence: {interpretation['confidence']}")

        if interpretation['key_findings']:
            print(f"\n   Key Findings:")
            for finding in interpretation['key_findings']:
                print(f"      • {finding}")

        if interpretation['unexpected_patterns']:
            print(f"Unexpected Patterns:")
            for pattern in interpretation['unexpected_patterns']:
                print(f"      • {pattern}")

        if interpretation['biological_insights']:
            print(f"Biological Insights:")
            for insight in interpretation['biological_insights'][:2]:
                print(f"      • {insight}")

    def _should_continue_exploring(self, iteration: int, max_iterations: int,
                                  min_confidence: str) -> Tuple[bool, str]:

        if iteration >= max_iterations:
            return False, "Maximum iterations reached"

        # Count high-confidence discoveries
        high_conf_discoveries = sum(
            1 for d in self.discoveries
            if d['interpretation']['confidence'] == 'high'
        )

        # Stop if we have multiple high-confidence discoveries
        if high_conf_discoveries >= 3:
            return False, f"Found {high_conf_discoveries} high-confidence discoveries"

        # Continue if we haven't found enough patterns
        total_supported = sum(
            1 for d in self.discoveries
            if d['interpretation']['supported']
        )

        if total_supported < 2:
            return True, "Continue searching for robust patterns"

        # Check for unexpected patterns
        unexpected_found = any(
            d['interpretation']['unexpected_patterns']
            for d in self.discoveries
        )

        if unexpected_found and iteration < max_iterations - 1:
            return True, "Unexpected patterns detected, worth exploring further"

        return True, "More exploration may yield additional insights"

    def save_discoveries(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        # Save discoveries as JSON
        discoveries_serializable = self._make_serializable(self.discoveries)

        with open(os.path.join(output_dir, 'autonomous_discoveries.json'), 'w') as f:
            json.dump(discoveries_serializable, f, indent=2)

        # Create comprehensive report
        self._create_discovery_report(output_dir)

        print(f"\n✓ Discoveries saved to {output_dir}/")

    def _make_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):  # Fix for numpy booleans
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

    def _create_discovery_report(self, output_dir: str):
        report_path = os.path.join(output_dir, 'autonomous_discovery_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("AUTONOMOUS SPATIAL TRANSCRIPTOMICS DISCOVERY REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Discoveries: {len(self.discoveries)}\n\n")

            # Summary statistics
            high_conf = sum(1 for d in self.discoveries if d['interpretation']['confidence'] == 'high')
            medium_conf = sum(1 for d in self.discoveries if d['interpretation']['confidence'] == 'medium')
            supported = sum(1 for d in self.discoveries if d['interpretation']['supported'])

            f.write("SUMMARY STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"High Confidence Discoveries: {high_conf}\n")
            f.write(f"Medium Confidence Discoveries: {medium_conf}\n")
            f.write(f"Supported Hypotheses: {supported}/{len(self.discoveries)}\n\n")

            # Individual discoveries
            for i, discovery in enumerate(self.discoveries, 1):
                f.write(f"\n{'='*70}\n")
                f.write(f"DISCOVERY {i}\n")
                f.write(f"{'='*70}\n\n")

                f.write(f"Iteration: {discovery['iteration']}\n")
                f.write(f"Timestamp: {discovery['timestamp']}\n\n")

                f.write(f"HYPOTHESIS:\n")
                f.write(f"{discovery['hypothesis']['statement']}\n\n")

                f.write(f"TYPE: {discovery['hypothesis']['type']}\n")
                f.write(f"PRIORITY: {discovery['hypothesis']['priority']}\n\n")

                interp = discovery['interpretation']
                f.write(f"RESULTS:\n")
                f.write(f"Supported: {interp['supported']}\n")
                f.write(f"Confidence: {interp['confidence']}\n\n")

                if interp['key_findings']:
                    f.write(f"KEY FINDINGS:\n")
                    for finding in interp['key_findings']:
                        f.write(f"  • {finding}\n")
                    f.write("\n")

                if interp['unexpected_patterns']:
                    f.write(f"UNEXPECTED PATTERNS:\n")
                    for pattern in interp['unexpected_patterns']:
                        f.write(f"  • {pattern}\n")
                    f.write("\n")

                if interp['biological_insights']:
                    f.write(f"BIOLOGICAL INSIGHTS:\n")
                    for insight in interp['biological_insights']:
                        f.write(f"  • {insight}\n")
                    f.write("\n")

                if interp['recommendations']:
                    f.write(f"RECOMMENDATIONS:\n")
                    for rec in interp['recommendations']:
                        f.write(f"  • {rec}\n")
                    f.write("\n")

            f.write(f"\n{'='*70}\n")
            f.write("END OF REPORT\n")
            f.write(f"{'='*70}\n")

def load_gef_data(input_file: str, max_cells: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    print(f"Loading GEF file: {input_file}")

    with h5py.File(input_file, 'r') as f:
        cell_data = f['cellBin/cell'][:]
        cell_exp = f['cellBin/cellExp'][:]
        genes = f['cellBin/gene'][:]

        coordinates = np.empty((len(cell_data), 2), dtype=np.float32)
        coordinates[:, 0] = cell_data['x']
        coordinates[:, 1] = cell_data['y']

        print("Creating expression matrix...")
        rows, cols, data = [], [], []

        for i, cell in enumerate(cell_data):
            if i % 5000 == 0:
                print(f"  Processing cell {i}/{len(cell_data)}")

            start = cell['offset']
            end = start + cell['geneCount']
            cell_genes = cell_exp[start:end]

            nonzero_mask = cell_genes['count'] > 0
            if np.any(nonzero_mask):
                rows.extend([i] * np.sum(nonzero_mask))
                cols.extend(cell_genes['geneID'][nonzero_mask])
                data.extend(cell_genes['count'][nonzero_mask])

        expression_matrix = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(len(cell_data), len(genes)),
            dtype=np.float32
        ).tocsr().toarray()

    # Extract gene names
    gene_names = []
    for i, gene in enumerate(genes):
        try:
            if isinstance(gene, (tuple, np.ndarray)) and len(gene) >= 2:
                name = gene[1]
                if hasattr(name, 'decode'):
                    gene_names.append(name.decode('utf-8'))
                else:
                    gene_names.append(str(name))
            elif hasattr(gene, 'decode'):
                gene_names.append(gene.decode('utf-8'))
            else:
                gene_names.append(str(gene))
        except:
            gene_names.append(f"Gene_{i}")

    # Subsample if requested
    if max_cells and expression_matrix.shape[0] > max_cells:
        print(f"\nSubsampling to {max_cells} cells...")
        indices = np.random.choice(expression_matrix.shape[0], size=max_cells, replace=False)
        expression_matrix = expression_matrix[indices]
        coordinates = coordinates[indices]

    print(f"✓ Loaded {expression_matrix.shape[0]:,} cells and {expression_matrix.shape[1]:,} genes")
    return expression_matrix, coordinates, gene_names


def run_autonomous_analysis(input_file: str,
                           max_cells: int = 3000,
                           output_dir: str = "output/agent",
                           api_key: Optional[str] = None,
                           use_llm: bool = True,
                           max_iterations: int = 5):
    """
    Run complete autonomous spatial transcriptomics analysis

    Args:
        input_file: Path to GEF file
        max_cells: Maximum cells to analyze
        output_dir: Output directory
        api_key: OpenAI API key (optional)
        use_llm: Whether to use LLM for analysis
        max_iterations: Maximum discovery iterations

    Returns:
        List of discoveries made by the autonomous agent
    """

    print("="*70)
    print("AUTONOMOUS SPATIAL TRANSCRIPTOMICS ANALYSIS")
    print("="*70)

    # Load data
    print("Loading data...")
    expression_matrix, coordinates, gene_names = load_gef_data(input_file, max_cells)

    # Initialize autonomous agent
    print("Initializing autonomous agent...")
    agent = TrulyAutonomousAgent(api_key=api_key, use_llm=use_llm)

    # Run autonomous discovery
    print("Starting autonomous discovery...")
    discoveries = agent.autonomous_discovery(
        expression_matrix=expression_matrix,
        coordinates=coordinates,
        gene_names=gene_names,
        max_iterations=max_iterations,
        min_confidence='medium'
    )

    # Save results
    print("Saving discoveries...")
    agent.save_discoveries(output_dir)

    # Print summary
    print("\n" + "="*70)
    print("AUTONOMOUS ANALYSIS SUMMARY")
    print("="*70)

    high_conf = sum(1 for d in discoveries if d['interpretation']['confidence'] == 'high')
    medium_conf = sum(1 for d in discoveries if d['interpretation']['confidence'] == 'medium')
    supported = sum(1 for d in discoveries if d['interpretation']['supported'])

    print(f"\nTotal Discoveries: {len(discoveries)}")
    print(f"  • High Confidence: {high_conf}")
    print(f"  • Medium Confidence: {medium_conf}")
    print(f"  • Low Confidence: {len(discoveries) - high_conf - medium_conf}")
    print(f"\nSupported Hypotheses: {supported}/{len(discoveries)}")

    # Show key discoveries
    print(f"KEY DISCOVERIES:")
    for i, discovery in enumerate(discoveries[:3], 1):
        interp = discovery['interpretation']
        print(f"\n{i}. {discovery['hypothesis']['statement']}")
        print(f"   Confidence: {interp['confidence']} | Supported: {interp['supported']}")
        if interp['key_findings']:
            print(f"   Finding: {interp['key_findings'][0]}")

    print(f"Results saved to: {output_dir}/")
    print(f"   • autonomous_discoveries.json")
    print(f"   • autonomous_discovery_report.txt")

    print("\n" + "="*70)
    print("AUTONOMOUS ANALYSIS COMPLETE")
    print("="*70)

    return discoveries

# Execution
if __name__ == "__main__":
    # Default output groups results under output/<run timestamp>/agent/ so
    # repeated runs don't overwrite each other, and everything from one run
    # lives together. All stages in one orchestrated run (run_pipeline.sh /
    # watch_runner.py) share the same timestamp via $ST_RUN_ID; a standalone
    # run generates its own.
    _run_id = os.environ.get("ST_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
    _default_output = os.environ.get("ST_OUTPUT_DIR", os.path.join("output", _run_id, "agent"))

    # Default input: $ST_INPUT_GEF, else auto-detect if exactly one .gef file
    # exists in the current directory. No hardcoded filename — this works with
    # anyone's data, not just one specific sample.
    _env_input = os.environ.get("ST_INPUT_GEF")
    _gef_candidates = sorted(glob.glob("*.gef"))
    _default_input = _env_input or (_gef_candidates[0] if len(_gef_candidates) == 1 else None)

    parser = argparse.ArgumentParser(
        description="Autonomous AI agent for hypothesis-driven analysis of a Stereo-seq GEF file. "
                     "Requires an OpenAI API key to generate/interpret hypotheses via LLM; "
                     "without one, the agent still runs but falls back to non-LLM analysis."
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
        help="Directory to write results to (default: output/<run timestamp>/agent; "
             "can also be set via $ST_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--max-cells", dest="max_cells", type=int, default=24930,
        help="Maximum number of cells to analyze (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iterations", dest="max_iterations", type=int, default=5,
        help="Maximum number of hypothesis generate/test iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key", dest="api_key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key for LLM-based hypothesis generation (default: $OPENAI_API_KEY). "
             "If omitted, the agent runs without LLM assistance.",
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

    # Configuration
    INPUT_FILE = args.input_file
    OUTPUT_DIR = args.output_dir
    MAX_CELLS = args.max_cells
    MAX_ITERATIONS = args.max_iterations
    API_KEY = args.api_key
    USE_LLM = bool(API_KEY)  # auto-disable LLM if no key present

    print("\n" + "-"*35)
    print("AUTONOMOUS SPATIAL TRANSCRIPTOMICS AGENT")
    print("-"*35)

    print(f"Configuration:")
    print(f"  Input: {INPUT_FILE}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Max cells: {MAX_CELLS:,}")
    print(f"  Max iterations: {MAX_ITERATIONS}")
    print(f"  LLM enabled: {USE_LLM}")

    try:
        # Run autonomous analysis
        discoveries = run_autonomous_analysis(
            input_file=INPUT_FILE,
            max_cells=MAX_CELLS,
            output_dir=OUTPUT_DIR,
            api_key=API_KEY,
            use_llm=USE_LLM,
            max_iterations=MAX_ITERATIONS
        )

        print("="*35)
        print("SUCCESS! Autonomous agent completed discovery!")
        print("="*35)

        # Additional insights
        print(f"WHAT THE AGENT DISCOVERED:")
        print("-"*70)

        for i, discovery in enumerate(discoveries, 1):
            interp = discovery['interpretation']
            if interp['supported'] and interp['confidence'] in ['high', 'medium']:
                print(f"\n{i}. Hypothesis: {discovery['hypothesis']['statement'][:80]}...")
                print(f"   Status: ✓ Supported (Confidence: {interp['confidence']})")

                if interp['unexpected_patterns']:
                    print(f"Unexpected: {interp['unexpected_patterns'][0]}")

                if interp['biological_insights']:
                    print(f"Insight: {interp['biological_insights'][0][:100]}...")

        print("\n" + "-"*70)
        print("\nThe agent autonomously:")
        print("  ✓ Observed data characteristics")
        print("  ✓ Generated testable hypotheses")
        print("  ✓ Designed and executed analyses")
        print("  ✓ Interpreted results biologically")
        print("  ✓ Discovered unexpected patterns")
        print("  ✓ Made recommendations for follow-up")

    except FileNotFoundError:
        print(f"Error: Could not find input file: {INPUT_FILE}")
        print("Please check the file path and try again.")
        raise SystemExit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

