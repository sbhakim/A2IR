# src/knowledge_graph/ftags/ftag_construct.py

import networkx as nx
import logging
import json
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class FuzzyTemporalAttackGraph:
    """
    Basic implementation of Fuzzy Temporal Attack Graphs (FTAGs).

    FTAGs extend traditional attack graphs by incorporating:
    - Fuzzy weights (μ) representing confidence in attack paths
    - Temporal constraints (τ) modeling attack evolution over time
    """

    def __init__(self, name: str = "a2ir_ftag"):
        """Initialize a new Fuzzy Temporal Attack Graph."""
        self.name = name
        self.graph = nx.DiGraph(name=name)
        logger.info(f"Initialized FTAG '{name}'")

    def add_node(self, node_id: str, node_type: str, attributes: Dict[str, Any] = None) -> bool:
        """
        Add a node to the FTAG.

        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (e.g., 'system_state', 'attack_step')
            attributes: Additional node attributes
        """
        if node_id in self.graph.nodes:
            return False

        # Prepare node attributes
        node_attrs = attributes or {}
        node_attrs['type'] = node_type
        node_attrs['created_at'] = datetime.now().timestamp()

        # Add the node to the graph
        self.graph.add_node(node_id, **node_attrs)
        logger.debug(f"Added node '{node_id}' of type '{node_type}'")
        return True

    def add_edge(self, source_id: str, target_id: str, confidence: float,
                 temporal_constraint: float, relation_type: str = "leads_to") -> bool:
        """
        Add an edge with fuzzy weight and temporal constraint.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            confidence: Fuzzy weight μ in [0, 1] representing confidence
            temporal_constraint: Temporal constraint τ in seconds
            relation_type: Type of relationship
        """
        # Ensure nodes exist
        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
            return False

        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        # Add the edge
        self.graph.add_edge(
            source_id,
            target_id,
            confidence=confidence,  # fuzzy weight μ
            temporal_constraint=temporal_constraint,  # temporal constraint τ
            relation_type=relation_type,
            created_at=datetime.now().timestamp()
        )
        logger.debug(f"Added edge '{source_id}' -> '{target_id}' with confidence {confidence:.2f}")
        return True

    def update_edge_confidence(self, source_id: str, target_id: str, new_confidence: float) -> bool:
        """Update an edge's confidence value."""
        if not self.graph.has_edge(source_id, target_id):
            return False

        self.graph[source_id][target_id]['confidence'] = max(0.0, min(1.0, new_confidence))
        self.graph[source_id][target_id]['updated_at'] = datetime.now().timestamp()
        return True

    def prune_low_confidence_edges(self, threshold: float = 0.2) -> int:
        """Remove edges with confidence below threshold."""
        edges_to_remove = []
        for u, v, data in self.graph.edges(data=True):
            if data['confidence'] < threshold:
                edges_to_remove.append((u, v))

        self.graph.remove_edges_from(edges_to_remove)
        logger.info(f"Pruned {len(edges_to_remove)} edges with confidence below {threshold}")
        return len(edges_to_remove)

    def find_attack_paths(self, source_node: str, target_node: str = None,
                          min_confidence: float = 0.5) -> List[List[str]]:
        """Find possible attack paths from source to target."""
        if source_node not in self.graph.nodes:
            return []

        if target_node and target_node not in self.graph.nodes:
            return []

        # Create a filtered graph with only edges above the confidence threshold
        filtered_graph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            if data['confidence'] >= min_confidence:
                filtered_graph.add_edge(u, v)

        # Find paths
        paths = []
        if target_node:
            try:
                for path in nx.all_simple_paths(filtered_graph, source_node, target_node):
                    paths.append(path)
            except nx.NetworkXNoPath:
                pass
        else:
            # If no target, find paths to all nodes
            for node in filtered_graph.nodes():
                if node == source_node:
                    continue
                try:
                    for path in nx.all_simple_paths(filtered_graph, source_node, node):
                        paths.append(path)
                except nx.NetworkXNoPath:
                    continue

        return paths

    def save_to_file(self, file_path: str) -> bool:
        """Save the FTAG to a JSON file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Convert graph to dictionary format
            graph_data = {
                "name": self.name,
                "nodes": [],
                "edges": []
            }

            # Add nodes
            for node_id, data in self.graph.nodes(data=True):
                node_data = data.copy()
                node_data['id'] = node_id
                graph_data['nodes'].append(node_data)

            # Add edges
            for source, target, data in self.graph.edges(data=True):
                edge_data = data.copy()
                edge_data['source'] = source
                edge_data['target'] = target
                graph_data['edges'].append(edge_data)

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(graph_data, f)

            logger.info(f"Saved FTAG to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save FTAG: {e}")
            return False

    @classmethod
    def load_from_file(cls, file_path: str) -> 'FuzzyTemporalAttackGraph':
        """Load an FTAG from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                graph_data = json.load(f)

            # Create new instance
            ftag = cls(name=graph_data.get('name', 'loaded_ftag'))

            # Add nodes
            for node_data in graph_data.get('nodes', []):
                node_id = node_data.pop('id')
                node_type = node_data.pop('type')
                ftag.add_node(node_id, node_type, node_data)

            # Add edges
            for edge_data in graph_data.get('edges', []):
                source = edge_data.pop('source')
                target = edge_data.pop('target')
                confidence = edge_data.pop('confidence')
                temporal_constraint = edge_data.pop('temporal_constraint')
                relation_type = edge_data.pop('relation_type', 'leads_to')
                ftag.add_edge(source, target, confidence, temporal_constraint, relation_type)

            logger.info(f"Loaded FTAG from {file_path}")
            return ftag

        except Exception as e:
            logger.error(f"Failed to load FTAG: {e}")
            return cls(name="fallback_ftag")  # Return empty graph as fallback