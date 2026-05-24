"""Evaluation: embedding extraction, probes, geometry, biology, baselines."""

from cellfm.eval.biology import opc_cycle_corr, stable_silhouette
from cellfm.eval.extract_embeddings import extract_embeddings
from cellfm.eval.geometry import knn_graph_jaccard, participation_ratio
from cellfm.eval.probes import knn_probe, linear_probe

__all__ = [
    "extract_embeddings",
    "linear_probe",
    "knn_probe",
    "participation_ratio",
    "knn_graph_jaccard",
    "stable_silhouette",
    "opc_cycle_corr",
]
