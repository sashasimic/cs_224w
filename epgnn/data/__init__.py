"""Shared data layer. Every model reads through here; no model file touches a parquet directly.

tables    load the pinned tables (zone_nodes_hourly, zone_edges_hourly, jao_cnec_hourly), cached
features  named feature sets and the gate-closure rule: the only place that says what day D may see
samples   (zone, delivery day) samples per split: history windows, day-D inputs, 24 targets + mask
scaling   asinh / residual-vs-last-week target transform, train-only standardisation
graph     per-day graphs: zone nodes + interconnector edges, optional constraint nodes + PTDF relations
"""
