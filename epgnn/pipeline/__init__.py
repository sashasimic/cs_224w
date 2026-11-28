"""Feature pipeline: from the pinned model tables (and the JAO views) to model-ready features.

    python -m epgnn.pipeline.build            # writes features/v1/ (about 5 minutes, ~1 GB on disk)

Modules
  registry        every feature column: group, source, whether it is known at gate closure, meaning
  zone_features   engineered per (zone, hour) columns on top of zone_nodes_hourly
  edge_features   engineered per (from_zone, to_zone, hour) columns on top of zone_edges_hourly
  constraints     JAO constraint nodes: stable ids, hourly RAM / Fmax / PTDF, per-day constraint graphs
  graphs          zone list, interconnector / full / self edge sets, Laplacian positional encoding
  tensors         dense (hour, zone, feature) arrays and the (delivery day, slot) index the torch models read

Nothing here edits the collaborators' modules; models opt in by reading features/v1/.
See docs/data_guide.md for what each source file holds and how it maps to these outputs.
"""
