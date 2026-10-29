"""
annotator.py
"""

import argparse
import glob
import h5py
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from scipy.stats import zscore, spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import warnings
import os
import time
import json

from .ovarian_knowledge_base import OvarianKnowledgeBase

warnings.filterwarnings('ignore')

# ── Hyper-parameters ─────────────────────────────────────────────────────────
TAU_Z               = 0.2    # min z-score to assign a cell           
TAU_D               = 2.5    # dominance ratio best/second-best        
TAU_R               = 0.3    # pathway correlation threshold         
N_MIN_DOMAIN        = 100    # min cells per functional domain         
HIGH_CONF_THRESHOLD = 0.7    # min confidence to count as high-confidence

TIER_WEIGHTS = {'core': 3.0, 'extended': 2.0, 'specialized': 1.5}


# =============================================================================
class EnhancedOvarianBiologyAnnotator:
    """
    Biological annotation using OvarianKnowledgeBase as the single source
    of truth for markers (8 cell types), pathways and developmental stages.
    """

    def __init__(self):
        self.kb = OvarianKnowledgeBase()

        # Runtime state
        self.detected_species:         Optional[str]    = None
        self.gene_mapping:             Dict[str, int]   = {}
        self._clean_gene_names:        List[str]        = []
        self.spatial_coherence_scores: Dict[str, float] = {}

    # =========================================================================
    # PHASE 1 – Gene Mapping 
    # =========================================================================
    def _build_gene_mapping(self, gene_names: List[str]) -> str:
        """
        Detect species from gene-name casing and build lookup
        Φ: variant → column_index.
        Handles GEF gene names stored as byte strings, tuples, or plain strings.
        """
        print("Performing species-adaptive gene recognition...")

        clean_names: List[str] = []
        for g in gene_names:
            gs = str(g).strip()
            if gs.startswith("(") and "b'" in gs:
                parts = [p.strip(" ()b'\"") for p in gs.split(",")]
                gs = parts[-1] if parts else gs
            elif gs.startswith("b'") or gs.startswith('b"'):
                gs = gs[2:].rstrip("'\"")
            clean_names.append(gs)

        self._clean_gene_names = clean_names

        sample  = clean_names[:min(1000, len(clean_names))]
        mouse_n = sum(1 for g in sample
                      if len(g) > 1 and g[0].isupper() and g[1:].islower())
        human_n = sum(1 for g in sample if g.isupper())
        mixed_n = len(sample) - mouse_n - human_n

        if mouse_n > human_n and mouse_n > mixed_n:
            species = 'mouse'
        elif human_n > mouse_n and human_n > mixed_n:
            species = 'human'
        else:
            species = 'mixed'

        print(f"Detected species: {species}")
        print(f"  Mouse patterns: {mouse_n}")
        print(f"  Human patterns: {human_n}")
        print(f"  Mixed patterns: {mixed_n}")

        gene_mapping: Dict[str, int] = {}
        for i, g in enumerate(clean_names):
            for var in [g, g.upper(), g.lower(), g.title(), g.capitalize()]:
                if var not in gene_mapping:
                    gene_mapping[var] = i

        self.gene_mapping     = gene_mapping
        self.detected_species = species
        return species

    def _resolve_markers(self, markers: List[str]) -> Tuple[List[int], List[str]]:
        """
        Resolve marker names → column indices via Φ.
        Falls back to partial substring match if exact lookup fails.
        """
        clean   = self._clean_gene_names
        indices: List[int] = []
        found:   List[str] = []

        for marker in markers:
            marker_found = False
            for var in [marker, marker.upper(), marker.lower()]:
                if var in self.gene_mapping:
                    idx = self.gene_mapping[var]
                    if idx not in indices:
                        indices.append(idx)
                        found.append(f"{marker}→{clean[idx]}")
                    marker_found = True
                    break

            if not marker_found:
                for i, g in enumerate(clean):
                    if marker.upper() in g.upper() or g.upper() in marker.upper():
                        if i not in indices:
                            indices.append(i)
                            found.append(f"{marker}≈{g}")
                        break

        return indices, found

    # =========================================================================
    # PHASE 2 – Cell Type Marker Scoring  
    # =========================================================================
    def _score_cell_types(
        self,
        expression_matrix: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """
        score_{t,c} = mean expression over detected markers in tier c  (line 30)
        combined    = weighted sum / total weight
        Normalise via MAD z-score, clip [-3, 3]                        (line 31)
        Uses self.kb.marker_genes (8 cell types from OvarianKnowledgeBase).
        """
        print("Calculating comprehensive cell type marker scores...")
        cell_type_analysis: Dict[str, Dict[str, Any]] = {}

        for cell_type, tiers in self.kb.marker_genes.items():
            print(f"  Analyzing {cell_type.replace('_', ' ').title()}...")

            weighted_scores: List[np.ndarray] = []
            total_weight                       = 0.0
            all_indices:     List[int]         = []
            all_found:       List[str]         = []

            for tier_name, weight in TIER_WEIGHTS.items():
                markers = tiers.get(tier_name, [])
                if not markers:
                    continue

                indices, found = self._resolve_markers(markers)
                if indices:
                    cat_score = np.mean(expression_matrix[:, indices], axis=1)
                    weighted_scores.append(cat_score * weight)
                    total_weight += weight
                    for idx in indices:
                        if idx not in all_indices:
                            all_indices.append(idx)
                    all_found.extend(found)

            # Weighted combination + MAD z-score normalisation
            if weighted_scores and total_weight > 0:
                composite = np.sum(weighted_scores, axis=0) / total_weight
                if np.std(composite) > 0:
                    median = np.median(composite)
                    mad    = np.median(np.abs(composite - median))
                    composite = ((composite - median) / (1.4826 * mad)
                                 if mad > 0 else zscore(composite))
                    composite = np.clip(composite, -3, 3)
            else:
                composite = np.zeros(expression_matrix.shape[0])

            all_flat = (tiers.get('core', []) +
                        tiers.get('extended', []) +
                        tiers.get('specialized', []))
            n_total  = len(all_flat)

            cell_type_analysis[cell_type] = {
                'composite_score':   composite,
                'available_markers': len(all_indices),
                'total_markers':     n_total,
                'marker_coverage':   len(all_indices) / max(n_total, 1),
                'found_genes':       all_found,
                'confidence_metrics': self._confidence_metrics(
                    all_indices, all_flat, composite
                ),
            }
            print(f"    Found {len(all_indices)}/{n_total} markers")

        return cell_type_analysis

    @staticmethod
    def _confidence_metrics(
        available_indices: List[int],
        all_markers: List[str],
        composite: np.ndarray,
    ) -> Dict[str, float]:
        coverage    = len(available_indices) / max(len(all_markers), 1)
        mean_s      = np.mean(composite)
        std_s       = np.std(composite)
        consistency = 1.0 / (1.0 + std_s / (abs(mean_s) + 1e-8))
        pos         = composite[composite > 0]
        signal      = float(np.mean(pos)) if len(pos) > 0 else 0.0
        return {
            'coverage_score':     coverage,
            'score_consistency':  consistency,
            'signal_strength':    signal,
            'overall_confidence': (coverage + consistency + min(signal, 1.0)) / 3,
        }

    # =========================================================================
    # PHASE 3 – Pathway Activity Scoring  
    # =========================================================================
    def _score_pathways(
        self,
        expression_matrix: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        act_p = exp( mean(log(X[:,g]+1)) ) − 1   geometric mean  (line 35)
        Z-score and clip to [-3, 3].
        Uses self.kb.pathways (8 flat pathways).
        """
        print("Phase 3: Pathway Activity Scoring...")
        activities: Dict[str, np.ndarray] = {}

        for pathway, genes in self.kb.pathways.items():
            indices, _ = self._resolve_markers(genes)
            if indices:
                expr = expression_matrix[:, indices]
                act  = np.exp(np.mean(np.log1p(expr), axis=1)) - 1
                if np.std(act) > 0:
                    act = zscore(act)
                    act = np.clip(act, -3, 3)
                activities[pathway] = act
                print(f"  {pathway}: {len(indices)} genes found")
            else:
                activities[pathway] = np.zeros(expression_matrix.shape[0])
                print(f"  {pathway}: 0 genes found")

        return activities

    # =========================================================================
    # PHASE 4 – Spatial Coherence 
    # =========================================================================
    def _spatial_coherence(
        self,
        coordinates: np.ndarray,
        ct_scores: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        coh_t = −SPEARMAN( D_spatial, D_score(z_t) )  (line 39)
        Samples up to 5,000 cells and 50,000 pairs for speed.
        """
        print("Phase 4: Spatial Coherence (Spearman)...")

        n     = coordinates.shape[0]
        n_s   = min(5_000, n)
        idx_s = (np.random.choice(n, size=n_s, replace=False)
                 if n > n_s else np.arange(n))

        try:
            D_sp = squareform(pdist(coordinates[idx_s])).flatten()
        except Exception as e:
            print(f"  Could not compute spatial distances: {e}")
            return {}

        coherence: Dict[str, float] = {}
        for ct, scores in ct_scores.items():
            ss = scores[idx_s]
            if np.std(ss) == 0:
                coherence[ct] = 0.0
                continue
            D_sc = squareform(pdist(ss.reshape(-1, 1))).flatten()
            np_  = len(D_sp)
            if np_ > 50_000:
                pi   = np.random.choice(np_, size=50_000, replace=False)
                r, _ = spearmanr(D_sp[pi], D_sc[pi])
            else:
                r, _ = spearmanr(D_sp, D_sc)
            coherence[ct] = -r if not np.isnan(r) else 0.0

        self.spatial_coherence_scores = coherence
        return coherence

    # =========================================================================
    # PHASE 5 – Biologically-Constrained Assignment  
    # =========================================================================
    def assign_cell_types_with_confidence(
        self,
        cell_type_analysis: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Line 42 – biological priors:
          • Oocyte scores < P95   → suppressed
          • Stromal fibroblasts   → boosted ×1.15
          • Immune cells < P85    → dampened ×0.9

        Line 43 – three-tier assignment gate:
          Primary  : max_score > TAU_Z  AND  dominance > TAU_D → high_confidence
          Secondary: max_score > 0.1    AND  dominance > 1.5   → dominance
          Tertiary : max_score > 0.005                          → low_confidence
          else     : unknown
        """
        print("Assigning cell types with confidence assessment...")

        ct_names = list(cell_type_analysis.keys())
        n_cells  = len(next(iter(cell_type_analysis.values()))['composite_score'])

        Z = np.column_stack([
            cell_type_analysis[ct]['composite_score'] for ct in ct_names
        ]).copy()

        # ── Line 42: Biological adjustments ──────────────────────────────
        def _col(name: str) -> Optional[int]:
            return ct_names.index(name) if name in ct_names else None

        if (oo := _col('oocytes')) is not None:
            Z[Z[:, oo] < np.percentile(Z[:, oo], 95), oo] = -np.inf
            print("  Applied: oocyte rarity suppression (P95)")

        if (sf := _col('stromal_fibroblasts')) is not None:
            Z[:, sf] *= 1.15
            print("  Applied: stromal fibroblast boost ×1.15")

        for im_ct in ('immune_macrophages', 'immune_tcells'):
            if (im := _col(im_ct)) is not None:
                Z[Z[:, im] < np.percentile(Z[:, im], 85), im] *= 0.9
        print("  Applied: immune cell dampening (P85)")

        # ── Line 43: Assignment loop ──────────────────────────────────────
        ct_arr   = np.full(n_cells, 'unknown', dtype='U30')
        conf_arr = np.zeros(n_cells)
        meth_arr = np.full(n_cells, 'unassigned', dtype='U20')

        for i in range(n_cells):
            row = Z[i, :]
            fin = np.isfinite(row)
            if not np.any(fin):
                continue

            best  = int(np.argmax(row))
            max_s = row[best]
            sv    = np.sort(row[fin])[::-1]
            dom   = sv[0] / (sv[1] + 1e-6) if len(sv) > 1 else float('inf')

            if max_s > TAU_Z and dom > TAU_D:         
                ct_arr[i]   = ct_names[best]
                conf_arr[i] = float(max_s)
                meth_arr[i] = 'high_confidence'
            elif max_s > 0.1 and dom > 1.5:          
                ct_arr[i]   = ct_names[best]
                conf_arr[i] = float(max_s) * 0.7
                meth_arr[i] = 'dominance'
            elif max_s > 0.005:                      
                ct_arr[i]   = ct_names[best]
                conf_arr[i] = float(max_s) * 0.5
                meth_arr[i] = 'low_confidence'

        assigned = int(np.sum(ct_arr != 'unknown'))
        print(f"  Assigned {assigned}/{n_cells} cells "
              f"({assigned / n_cells * 100:.1f}%)")

        stats = {
            'total_cells':       n_cells,
            'assigned_cells':    assigned,
            'high_confidence':   int(np.sum(meth_arr == 'high_confidence')),
            'medium_confidence': int(np.sum(meth_arr == 'dominance')),
            'low_confidence':    int(np.sum(meth_arr == 'low_confidence')),
            'unassigned':        int(n_cells - assigned),
            'assignment_rate':   assigned / n_cells,
        }
        return {
            'assignments': {
                'cell_type':         ct_arr,
                'primary_score':     conf_arr,
                'confidence':        conf_arr,
                'assignment_method': meth_arr,
            },
            'statistics':       stats,
            'cell_type_counts': Counter(ct_arr),
        }

    # =========================================================================
    # PHASE 6 – Pathway Co-expression Network 
    # =========================================================================
    def _pathway_coexpression(
        self,
        pathway_activities: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        R ← { (p1,p2,r) : r=PEARSON(act_p1, act_p2), |r|>τ_r }
        """
        print(f"Phase 6: Pathway Co-expression Network (τ_r={TAU_R})...")
        names  = [k for k, v in pathway_activities.items() if np.std(v) > 0]
        result: Dict[str, float] = {}

        for i, p1 in enumerate(names):
            for p2 in names[i + 1:]:
                r = float(np.corrcoef(
                    pathway_activities[p1], pathway_activities[p2]
                )[0, 1])
                if not np.isnan(r) and abs(r) > TAU_R:
                    result[f"{p1}--{p2}"] = r

        print(f"  Found {len(result)} significant correlations")
        return result

    # =========================================================================
    # PHASE 7 – Functional Domain Detection 
    # =========================================================================
    def _functional_domains(
        self,
        coordinates: np.ndarray,
        pathway_activities: Dict[str, np.ndarray],
    ) -> Dict[str, Dict]:
        """
        H_p = { i : act_p[i] > P80 }
        DBSCAN(ε=100 µm) on spatial coords of H_p cells.
        """
        print(f"Phase 7: Functional Domain Detection "
              f"(n_min={N_MIN_DOMAIN}, ε=100 µm)...")
        domains: Dict[str, Dict] = {}

        for pathway, acts in pathway_activities.items():
            if np.std(acts) == 0:
                continue
            H_p = np.where(acts > np.percentile(acts, 80))[0]
            if len(H_p) < N_MIN_DOMAIN:
                continue

            labels = DBSCAN(
                eps=100, min_samples=N_MIN_DOMAIN // 2
            ).fit_predict(coordinates[H_p])

            for cid in np.unique(labels):
                if cid == -1:
                    continue
                cells = H_p[labels == cid]
                if len(cells) < N_MIN_DOMAIN:
                    continue
                domains[f"{pathway}_domain_{cid}"] = {
                    'pathway':        pathway,
                    'cells':          cells,
                    'center':         np.mean(coordinates[cells], axis=0),
                    'activity_level': float(np.mean(acts[cells])),
                    'size':           len(cells),
                }

        print(f"  Total domains found: {len(domains)}")
        return domains

    # =========================================================================
    # SUPPLEMENTARY ANALYSES
    # =========================================================================
    def analyze_spatial_organization_enhanced(
        self,
        coordinates: np.ndarray,
        cell_assignments: Dict[str, Any],
    ) -> Dict[str, Any]:
        print("Performing enhanced spatial organization analysis...")
        cts         = cell_assignments['assignments']['cell_type']
        n_neighbors = min(20, len(coordinates) // 10)
        nbrs        = NearestNeighbors(n_neighbors=n_neighbors).fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)

        result: Dict[str, Any] = {
            'cell_type_distributions': {},
            'neighborhood_analysis':   [],
            'global_patterns':         {},
        }

        for ct in np.unique(cts):
            if ct == 'unknown':
                continue
            mask   = cts == ct
            coords = coordinates[mask]
            if len(coords) < 5:
                continue
            cen = np.mean(coords, axis=0)
            result['cell_type_distributions'][ct] = {
                'count':          int(np.sum(mask)),
                'centroid':       cen.tolist(),
                'spread':         np.std(coords, axis=0).tolist(),
                'compactness':    float(np.mean(np.linalg.norm(coords - cen, axis=1))),
                'spatial_extent': {
                    'x_range': (float(coords[:, 0].min()), float(coords[:, 0].max())),
                    'y_range': (float(coords[:, 1].min()), float(coords[:, 1].max())),
                },
            }

        for i in range(len(coordinates)):
            nbr_cts = cts[indices[i][1:]]
            cnts    = Counter(nbr_cts)
            total   = len(indices[i]) - 1
            result['neighborhood_analysis'].append({
                'cell_id':   i,
                'cell_type': cts[i],
                'neighborhood_composition': {ct: c / total for ct, c in cnts.items()},
                'local_density': float(np.mean(1.0 / (distances[i][1:] + 1e-8))),
            })

        mixing = []
        for i in range(len(coordinates)):
            cnts = Counter(cts[indices[i][1:]])
            if len(cnts) > 1:
                t = sum(cnts.values())
                mixing.append(-sum((c / t) * np.log(c / t) for c in cnts.values()))
        result['global_patterns']['spatial_mixing_index'] = (
            float(np.mean(mixing)) if mixing else 0.0
        )
        return result

    def identify_anatomical_structures_enhanced(
        self,
        coordinates: np.ndarray,
        cell_assignments: Dict[str, Any],
        cell_type_analysis: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        print("Identifying anatomical structures...")
        found: Dict[str, Any] = {}
        cts   = cell_assignments['assignments']['cell_type']

        anatomical_defs = {
            'primordial_follicles': ['Foxl2', 'FOXL2', 'Amh', 'AMH', 'Nobox', 'NOBOX'],
            'primary_follicles':    ['Foxl2', 'FOXL2', 'Fshr', 'FSHR', 'Kit', 'KIT'],
            'secondary_follicles':  ['Fshr', 'FSHR', 'Cyp19a1', 'CYP19A1', 'Inha', 'INHA'],
            'antral_follicles':     ['Fshr', 'FSHR', 'Cyp19a1', 'CYP19A1', 'Lhcgr', 'LHCGR'],
            'corpus_luteum_struct': ['Star', 'STAR', 'Cyp11a1', 'CYP11A1', 'Hsd3b1', 'HSD3B1'],
        }

        for struct_name, struct_markers in anatomical_defs.items():
            scores = np.zeros(len(cts))
            for marker in struct_markers:
                for ct, analysis in cell_type_analysis.items():
                    if any(marker in g for g in analysis.get('found_genes', [])):
                        mask = cts == ct
                        scores[mask] += analysis['composite_score'][mask]

            if np.std(scores) > 0:
                sz    = (scores - np.mean(scores)) / np.std(scores)
                hmask = sz > 1.0
                if np.sum(hmask) > 0:
                    hc = coordinates[hmask]
                    found[struct_name] = {
                        'cell_count':    int(np.sum(hmask)),
                        'centroid':      np.mean(hc, axis=0).tolist(),
                        'spatial_extent': {
                            'x_range': (float(hc[:, 0].min()), float(hc[:, 0].max())),
                            'y_range': (float(hc[:, 1].min()), float(hc[:, 1].max())),
                        },
                        'average_score': float(np.mean(sz[hmask])),
                        'confidence':    float(np.mean(sz[hmask]) / 3.0),
                    }
        return found

    def perform_developmental_stage_analysis(
        self,
        expression_matrix: np.ndarray,
        cell_assignments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Uses self.kb.developmental_stages (6 stages) directly."""
        print("Performing developmental stage analysis...")
        result: Dict[str, Any] = {}

        for stage, markers in self.kb.developmental_stages.items():
            indices, found = self._resolve_markers(markers)
            if indices:
                scores = np.mean(expression_matrix[:, indices], axis=1)
                if np.std(scores) > 0:
                    scores = zscore(scores)
                    scores = np.clip(scores, -3, 3)
                thresh = np.percentile(scores, 85)
                active = scores > thresh
                result[stage] = {
                    'stage_scores':    scores,
                    'active_cells':    active,
                    'n_active_cells':  int(np.sum(active)),
                    'markers_found':   found,
                    'marker_coverage': len(found) / max(len(markers), 1),
                }
        return result

    # =========================================================================
    # PIPELINE
    # =========================================================================
    def run_complete_enhanced_annotation(
        self,
        expression_matrix: np.ndarray,
        coordinates: np.ndarray,
        gene_names: List[str],
    ) -> Dict[str, Any]:
        print("Starting enhanced biological annotation pipeline...")

        # Phase 1
        self._build_gene_mapping(gene_names)

        # Phase 2
        cell_type_analysis = self._score_cell_types(expression_matrix)

        # Phase 3
        pathway_activities = self._score_pathways(expression_matrix)

        # Phase 4
        ct_scores_only = {ct: d['composite_score'] for ct, d in cell_type_analysis.items()}
        spatial_coherence = self._spatial_coherence(coordinates, ct_scores_only)

        # Phase 5
        cell_assignments = self.assign_cell_types_with_confidence(cell_type_analysis)

        # Phase 6
        pathway_corr = self._pathway_coexpression(pathway_activities)

        # Phase 7
        func_domains = self._functional_domains(coordinates, pathway_activities)

        # Supplementary
        spatial_analysis      = self.analyze_spatial_organization_enhanced(
            coordinates, cell_assignments
        )
        anatomical_structures = self.identify_anatomical_structures_enhanced(
            coordinates, cell_assignments, cell_type_analysis
        )
        dev_analysis = self.perform_developmental_stage_analysis(
            expression_matrix, cell_assignments
        )

        # ── Statistics ────────────────────────────────────────────────────
        stats         = cell_assignments['statistics']
        assigned_mask = cell_assignments['assignments']['cell_type'] != 'unknown'
        hc_count      = int(np.sum(
            cell_assignments['assignments']['confidence'][assigned_mask] > HIGH_CONF_THRESHOLD
        ))
        hc_rate = (hc_count / stats['assigned_cells']
                   if stats['assigned_cells'] > 0 else 0.0)

        results = {
            'cell_type_analysis':     cell_type_analysis,
            'cell_assignments':       cell_assignments,
            'spatial_analysis':       spatial_analysis,
            'anatomical_structures':  anatomical_structures,
            'developmental_analysis': dev_analysis,
            'pathway_correlations':   pathway_corr,
            'functional_domains':     func_domains,
            'spatial_coherence':      spatial_coherence,
            'species_detected':       self.detected_species,
            'annotation_statistics':  {
                'total_cells':           stats['total_cells'],
                'annotated_cells':       stats['assigned_cells'],
                'annotation_rate':       stats['assignment_rate'],
                'high_confidence_count': hc_count,
                'high_confidence_rate':  hc_rate,
                'cell_type_counts':      dict(cell_assignments['cell_type_counts']),
                'average_confidence':    float(
                    np.mean(cell_assignments['assignments']['confidence'])
                ),
                'marker_genes_detected': sum(
                    d['available_markers'] for d in cell_type_analysis.values()
                ),
                'total_marker_genes':    sum(
                    d['total_markers'] for d in cell_type_analysis.values()
                ),
                'pathway_correlations':  len(pathway_corr),
                'functional_domains':    len(func_domains),
            },
        }

        ann_s = results['annotation_statistics']
        print(f"Enhanced annotation complete:")
        print(f"  Success rate: {ann_s['annotation_rate']:.1%}")
        print(f"  High confidence (z>{HIGH_CONF_THRESHOLD}): {hc_rate:.1%}")
        print(f"  Species detected: {self.detected_species}")
        print(f"  Marker genes found: "
              f"{ann_s['marker_genes_detected']}/{ann_s['total_marker_genes']}")
        return results

    # =========================================================================
    # VISUALISATIONS
    # =========================================================================
    def create_comprehensive_visualizations(
        self,
        coordinates: np.ndarray,
        annotation_results: Dict[str, Any],
        output_dir: str,
    ):
        print("Creating comprehensive annotation visualizations...")
        os.makedirs(output_dir, exist_ok=True)
        self._plot_cell_type_assignments(coordinates, annotation_results, output_dir)
        self._plot_confidence_analysis(coordinates, annotation_results, output_dir)
        self._plot_spatial_organization(coordinates, annotation_results, output_dir)
        self._plot_anatomical_structures(coordinates, annotation_results, output_dir)
        self._plot_annotation_overview(coordinates, annotation_results, output_dir)
        print(f"Visualizations saved to {output_dir}")

    def _plot_cell_type_assignments(self, coordinates, r, output_dir):
        fig, ax  = plt.subplots(figsize=(12, 10))
        cts      = r['cell_assignments']['assignments']['cell_type']
        unique   = np.unique(cts)
        colors   = plt.cm.Set3(np.linspace(0, 1, len(unique)))
        for i, ct in enumerate(unique):
            mask = cts == ct
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1],
                       c=[colors[i]], label=ct.replace('_', ' ').title(),
                       s=3, alpha=0.7)
        ax.set_title('Cell Type Assignments')
        ax.set_xlabel('Spatial X'); ax.set_ylabel('Spatial Y')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cell_type_assignments.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confidence_analysis(self, coordinates, r, output_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        conf = r['cell_assignments']['assignments']['confidence']
        sc   = ax1.scatter(coordinates[:, 0], coordinates[:, 1],
                           c=conf, cmap='viridis', s=3, alpha=0.7)
        ax1.set_title('Assignment Confidence (Spatial)')
        plt.colorbar(sc, ax=ax1)
        ax2.hist(conf, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(conf), color='red', linestyle='--',
                    label=f'Mean: {np.mean(conf):.3f}')
        ax2.set_title('Confidence Score Distribution'); ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_spatial_organization(self, coordinates, r, output_dir):
        sa        = r['spatial_analysis']
        densities = [n['local_density'] for n in sa['neighborhood_analysis']]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sc = ax1.scatter(coordinates[:, 0], coordinates[:, 1],
                         c=densities, cmap='coolwarm', s=3, alpha=0.7)
        ax1.set_title('Local Cell Density'); plt.colorbar(sc, ax=ax1)
        for ct, info in sa['cell_type_distributions'].items():
            c = info['centroid']
            ax2.scatter(c[0], c[1], s=100, label=ct.replace('_', ' ').title())
        ax2.set_title('Cell Type Centroids'); ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spatial_organization.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_anatomical_structures(self, coordinates, r, output_dir):
        structures = r['anatomical_structures']
        if not structures:
            print("No anatomical structures to plot"); return
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(coordinates[:, 0], coordinates[:, 1],
                   c='lightgray', s=1, alpha=0.3)
        colors = plt.cm.tab10(np.linspace(0, 1, len(structures)))
        for i, (name, data) in enumerate(structures.items()):
            c = data['centroid']
            ax.scatter(c[0], c[1], c=[colors[i]], s=200,
                       label=f"{name.replace('_', ' ').title()} ({data['cell_count']} cells)",
                       marker='*', edgecolors='black', linewidth=1)
        ax.set_title('Identified Anatomical Structures'); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'anatomical_structures.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_annotation_overview(self, coordinates, r, output_dir):
        fig = plt.figure(figsize=(20, 12))
        gs  = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        ca    = r['cell_assignments']
        cts   = ca['assignments']['cell_type']
        conf  = ca['assignments']['confidence']
        stats = ca['statistics']
        ann_s = r['annotation_statistics']

        ax1 = fig.add_subplot(gs[0, :2])
        unique = np.unique(cts)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
        for i, ct in enumerate(unique):
            if ct == 'unknown': continue
            mask = cts == ct
            ax1.scatter(coordinates[mask, 0], coordinates[mask, 1],
                        c=[colors[i]], label=ct.replace('_', ' ').title(),
                        s=2, alpha=0.7)
        ax1.set_title('Cell Type Spatial Distribution')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        ax2 = fig.add_subplot(gs[0, 2])
        valid = {k: v for k, v in ca['cell_type_counts'].items() if k != 'unknown'}
        if valid:
            ax2.pie(valid.values(),
                    labels=[k.replace('_', ' ').title() for k in valid],
                    autopct='%1.1f%%', startangle=90)
            ax2.set_title('Cell Type Proportions')

        ax3 = fig.add_subplot(gs[0, 3])
        qm   = ['High Conf.', 'Med. Conf.', 'Low Conf.', 'Unassigned']
        qv   = [stats['high_confidence'], stats['medium_confidence'],
                stats['low_confidence'],  stats['unassigned']]
        bars = ax3.bar(qm, qv, color=['green', 'orange', 'yellow', 'red'], alpha=0.7)
        ax3.set_title('Assignment Quality'); ax3.set_ylabel('Number of Cells')
        for b, v in zip(bars, qv):
            ax3.text(b.get_x() + b.get_width() / 2.,
                     b.get_height() + max(qv) * 0.01,
                     f'{v}', ha='center', va='bottom')

        ax4 = fig.add_subplot(gs[1, :2])
        sc  = ax4.scatter(coordinates[:, 0], coordinates[:, 1],
                          c=conf, cmap='viridis', s=2, alpha=0.7)
        ax4.set_title('Confidence Score Distribution')
        plt.colorbar(sc, ax=ax4)

        ax5 = fig.add_subplot(gs[1, 2:])
        cta      = r['cell_type_analysis']
        ct_names = list(cta.keys())
        fa  = [cta[ct]['available_markers'] for ct in ct_names]
        ma  = [cta[ct]['total_markers'] - cta[ct]['available_markers'] for ct in ct_names]
        x   = np.arange(len(ct_names))
        ax5.bar(x, fa, label='Found',   color='green', alpha=0.7)
        ax5.bar(x, ma, label='Missing', color='red',   alpha=0.7, bottom=fa)
        ax5.set_xticks(x)
        ax5.set_xticklabels([ct.replace('_', ' ').title() for ct in ct_names],
                            rotation=45, ha='right')
        ax5.set_title('Marker Gene Detection by Cell Type'); ax5.legend()

        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        summary = (
            f"ANNOTATION SUMMARY STATISTICS\n\n"
            f"Total Cells Analyzed: {stats['total_cells']:,}\n"
            f"Successfully Annotated: {stats['assigned_cells']:,} "
            f"({stats['assignment_rate']:.1%})\n"
            f"High Confidence Assignments: {stats['high_confidence']:,}\n"
            f"High-Conf Rate (z>{HIGH_CONF_THRESHOLD}): {ann_s['high_confidence_rate']:.1%}\n"
            f"Species Detected: {r.get('species_detected', '?')}\n\n"
            f"Cell Types Identified: "
            f"{len([ct for ct in ca['cell_type_counts'] if ct != 'unknown'])}\n"
            f"Marker genes: {ann_s['marker_genes_detected']}/{ann_s['total_marker_genes']}\n"
            f"Pathway correlations: {ann_s['pathway_correlations']}\n"
            f"Functional domains: {ann_s['functional_domains']}"
        )
        if r.get('anatomical_structures'):
            summary += f"\nAnatomical Structures Found: {len(r['anatomical_structures'])}"
        if r.get('spatial_analysis', {}).get('global_patterns'):
            mi = r['spatial_analysis']['global_patterns'].get('spatial_mixing_index', 0)
            summary += f"\nSpatial Mixing Index: {mi:.3f}"
        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=12,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.suptitle('Comprehensive Biological Annotation Overview',
                     fontsize=16, y=0.98)
        plt.savefig(os.path.join(output_dir, 'annotation_overview.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


# =============================================================================
# DATA LOADER
# =============================================================================
class RealDataLoader:
    def process_gef_file(
        self, input_file: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        print("Reading GEF file...")
        with h5py.File(input_file, 'r') as f:
            cell_data = f['cellBin/cell'][:]
            cell_exp  = f['cellBin/cellExp'][:]
            genes     = f['cellBin/gene'][:]

            coordinates       = np.empty((len(cell_data), 2), dtype=np.float32)
            coordinates[:, 0] = cell_data['x']
            coordinates[:, 1] = cell_data['y']

            print("Creating expression matrix...")
            rows, cols, data = [], [], []
            for i, cell in enumerate(cell_data):
                if i % 1000 == 0:
                    print(f"Processing cell {i}/{len(cell_data)}")
                start = cell['offset']
                end   = start + cell['geneCount']
                cg    = cell_exp[start:end]
                nz    = cg['count'] > 0
                if np.any(nz):
                    rows.extend([i] * int(np.sum(nz)))
                    cols.extend(cg['geneID'][nz])
                    data.extend(cg['count'][nz])

            expression_matrix = sparse.coo_matrix(
                (data, (rows, cols)),
                shape=(len(cell_data), len(genes)),
                dtype=np.float32,
            ).tocsr().toarray()

        # Extract clean gene names from the GEF structured dtype
        gene_names: List[str] = []
        for gene in genes:
            try:
                if hasattr(gene, 'dtype') and 'geneName' in gene.dtype.names:
                    raw = gene['geneName']
                elif hasattr(gene, '__len__') and len(gene) >= 2:
                    raw = gene[1]
                else:
                    raw = gene
                name = raw.decode('utf-8') if isinstance(raw, (bytes, np.bytes_)) else str(raw)
            except Exception:
                name = str(gene)
            gene_names.append(name.strip())

        print(f"Loaded {expression_matrix.shape[0]:,} cells and "
              f"{expression_matrix.shape[1]:,} genes")
        print(f"Coordinate range: "
              f"X({coordinates[:,0].min():.1f} to {coordinates[:,0].max():.1f}), "
              f"Y({coordinates[:,1].min():.1f} to {coordinates[:,1].max():.1f})")
        return expression_matrix, coordinates, gene_names


# =============================================================================
# ENTRY POINT
# =============================================================================
def run_enhanced_real_data_annotation(
    input_file: str,
    max_cells:  Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:

    print("=" * 70)
    print("OVARIAN TISSUE BIOLOGICAL ANNOTATION")
    print("=" * 70)
    print(f"Input file: {input_file}")
    if max_cells:
        print(f"Cell limit: {max_cells} (for testing)")

    start_time = time.time()

    print("\n" + "=" * 50)
    print("STEP 1: LOADING REAL GEF DATA")
    print("=" * 50)
    loader = RealDataLoader()
    expression_matrix, coordinates, gene_names = loader.process_gef_file(input_file)

    if max_cells and expression_matrix.shape[0] > max_cells:
        print(f"\nLimiting analysis to {max_cells} cells for testing...")
        idx = np.random.choice(expression_matrix.shape[0], size=max_cells, replace=False)
        expression_matrix = expression_matrix[idx]
        coordinates       = coordinates[idx]
        print(f"Dataset reduced to {expression_matrix.shape[0]} cells")

    print("\n" + "=" * 50)
    print("STEP 2: BIOLOGICAL ANNOTATION")
    print("=" * 50)
    annotator = EnhancedOvarianBiologyAnnotator()
    results   = annotator.run_complete_enhanced_annotation(
        expression_matrix, coordinates, gene_names
    )

    if output_dir:
        print("\n" + "=" * 50)
        print("STEP 3: CREATING VISUALIZATIONS")
        print("=" * 50)
        annotator.create_comprehensive_visualizations(coordinates, results, output_dir)

    print("\n" + "=" * 50)
    print("STEP 4: ENHANCED ANNOTATION RESULTS")
    print("=" * 50)
    ann_s = results['annotation_statistics']
    stats = results['cell_assignments']['statistics']

    print(f"\nDataset Summary:")
    print(f"  Total cells analyzed: {ann_s['total_cells']:,}")
    print(f"  Successfully annotated: {ann_s['annotated_cells']:,}")
    print(f"  Annotation success rate: {ann_s['annotation_rate']:.1%}")
    print(f"  Average confidence score: {ann_s['average_confidence']:.3f}")
    print(f"  Species detected: {results['species_detected']}")
    print(f"  Marker genes detected: "
          f"{ann_s['marker_genes_detected']}/{ann_s['total_marker_genes']}")
    print(f"\n  [Phase 6] Pathway correlations:  {ann_s['pathway_correlations']}")
    print(f"  [Phase 7] Functional domains:    {ann_s['functional_domains']}")
    print(f"  High-confidence rate (z>{HIGH_CONF_THRESHOLD}):    {ann_s['high_confidence_rate']:.1%}")

    print(f"\nCell Type Distribution:")
    for ct, cnt in sorted(ann_s['cell_type_counts'].items(),
                          key=lambda x: x[1], reverse=True):
        pct = cnt / ann_s['total_cells'] * 100
        print(f"  {ct.replace('_', ' ').title()}: {cnt:,} cells ({pct:.1f}%)")

    if output_dir:
        print("\n" + "=" * 50)
        print("STEP 5: SAVING RESULTS")
        print("=" * 50)
        os.makedirs(output_dir, exist_ok=True)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):  return obj.tolist()
                if isinstance(obj, np.integer):  return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                return super().default(obj)

        rf = os.path.join(output_dir, 'enhanced_annotation_results.json')
        with open(rf, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        rep = os.path.join(output_dir, 'enhanced_annotation_report.txt')
        with open(rep, 'w', encoding='utf-8') as f:
            f.write("ENHANCED OVARIAN TISSUE BIOLOGICAL ANNOTATION REPORT\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input File: {input_file}\n")
            f.write(f"Cells Analyzed: {ann_s['total_cells']:,}\n")
            f.write(f"Species Detected: {results['species_detected']}\n")
            f.write(f"Annotation Success Rate: {ann_s['annotation_rate']:.1%}\n")
            f.write(f"High-Confidence Rate (z>{HIGH_CONF_THRESHOLD}): "
                    f"{ann_s['high_confidence_rate']:.1%}\n")
            f.write(f"Marker Genes Found: "
                    f"{ann_s['marker_genes_detected']}/{ann_s['total_marker_genes']}\n\n")

            f.write("CELL TYPE DISTRIBUTION\n" + "-" * 25 + "\n")
            for ct, cnt in sorted(ann_s['cell_type_counts'].items(),
                                  key=lambda x: x[1], reverse=True):
                pct = cnt / ann_s['total_cells'] * 100
                f.write(f"{ct.replace('_', ' ').title()}: {cnt:,} cells ({pct:.1f}%)\n")

            f.write("\nANATOMICAL STRUCTURES\n" + "-" * 20 + "\n")
            structs = results.get('anatomical_structures', {})
            if structs:
                for name, data in structs.items():
                    f.write(f"{name.replace('_', ' ').title()}:\n")
                    f.write(f"  Cells: {data['cell_count']:,}\n")
                    f.write(f"  Confidence: {data.get('confidence', 0):.3f}\n")
                    f.write(f"  Location: ({data['centroid'][0]:.1f}, "
                            f"{data['centroid'][1]:.1f})\n\n")
            else:
                f.write("No anatomical structures detected above threshold\n")

            f.write("\nDEVELOPMENTAL STAGES\n" + "-" * 19 + "\n")
            for stage, sd in results.get('developmental_analysis', {}).items():
                pct = sd['n_active_cells'] / ann_s['total_cells'] * 100
                f.write(f"{stage.replace('_', ' ').title()}:\n")
                f.write(f"  Active cells: {sd['n_active_cells']:,} ({pct:.1f}%)\n")
                f.write(f"  Marker coverage: {sd['marker_coverage']:.1%}\n\n")

            f.write("\nPHASE 6 – PATHWAY CO-EXPRESSION\n" + "-" * 30 + "\n")
            for pair, rv in results.get('pathway_correlations', {}).items():
                f.write(f"  {pair}: r = {rv:.3f}\n")
            if not results.get('pathway_correlations'):
                f.write("  No significant correlations found\n")

            f.write(f"\nPHASE 7 – FUNCTIONAL DOMAINS\n" + "-" * 28 + "\n")
            for dname, dd in results.get('functional_domains', {}).items():
                f.write(f"  {dname}: {dd['size']} cells, "
                        f"activity={dd['activity_level']:.3f}\n")
            if not results.get('functional_domains'):
                f.write("  No functional domains detected\n")

        print(f"Results saved to:")
        print(f"  - {rf}")
        print(f"  - {rep}")

    elapsed = time.time() - start_time
    print(f"\n" + "=" * 70)
    print("ENHANCED ANNOTATION ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"Total processing time: {elapsed:.2f} seconds")
    print(f"Cells processed per second: {ann_s['total_cells'] / elapsed:.1f}")
    return results


if __name__ == "__main__":
    # Default output groups results under output/<run timestamp>/annotator/ so
    # repeated runs don't overwrite each other, and everything from one run
    # lives together. All stages in one orchestrated run (run_pipeline.sh /
    # watch_runner.py) share the same timestamp via $ST_RUN_ID; a standalone
    # run generates its own.
    _run_id = os.environ.get("ST_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")
    _default_output = os.environ.get("ST_OUTPUT_DIR", os.path.join("output", _run_id, "annotator"))

    # Default input: $ST_INPUT_GEF, else auto-detect if exactly one .gef file
    # exists in the current directory. No hardcoded filename — this works with
    # anyone's data, not just one specific sample.
    _env_input = os.environ.get("ST_INPUT_GEF")
    _gef_candidates = sorted(glob.glob("*.gef"))
    _default_input = _env_input or (_gef_candidates[0] if len(_gef_candidates) == 1 else None)

    parser = argparse.ArgumentParser(
        description="Biology-guided cell type annotation on a Stereo-seq GEF file."
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
        help="Directory to write results to (default: output/<run timestamp>/annotator; "
             "can also be set via $ST_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--max-cells", dest="max_cells", type=int, default=None,
        help="Optional cap on the number of cells to process (default: all cells)",
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
    input_file            = args.input_file
    output_dir            = args.output_dir
    max_cells_for_testing = args.max_cells

    print("Enhanced Ovarian Biology Annotator - GEF Data Analysis")
    print("=" * 55)

    try:
        print("\nRunning enhanced annotation on GEF data...")
        results = run_enhanced_real_data_annotation(
            input_file=input_file,
            max_cells=max_cells_for_testing,
            output_dir=output_dir,
        )
        if results:
            print("\nGEF data annotation completed successfully!")
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except FileNotFoundError:
        print(f"\nError: Could not find input file: {input_file}")
        print("Please check the file path and try again.")
        raise SystemExit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
