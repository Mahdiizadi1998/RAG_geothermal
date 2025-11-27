"""
Knowledge Graph - Document relationship graph for multi-hop reasoning
Uses NetworkX to connect documents based on similarity and metadata
"""

import logging
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Document-level knowledge graph for relationship-based retrieval
    
    Nodes: Documents or document sections
    Edges: Semantic similarity (>threshold) or metadata connections
    
    Enables:
    - Multi-hop reasoning (traverse from Doc A to related Doc B)
    - Relationship queries (find all docs connected to Doc X)
    - Cluster analysis (find document communities)
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7,
                 metadata_edge_types: List[str] = None):
        """
        Initialize knowledge graph
        
        Args:
            embedding_model: Model for computing similarities
            similarity_threshold: Minimum similarity for edges (0-1)
            metadata_edge_types: Types of metadata connections to create
        """
        self.similarity_threshold = similarity_threshold
        self.metadata_edge_types = metadata_edge_types or ['same_well', 'same_operator', 'same_formation']
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize graph
        self.graph = nx.Graph()
        
        # Node storage
        self.node_embeddings = {}  # node_id -> embedding
        self.node_data = {}  # node_id -> full data dict
    
    def build_graph(self, chunks: List[Dict]) -> Dict:
        """
        Build knowledge graph from document chunks
        
        Args:
            chunks: List of chunk dicts with 'text', 'chunk_id', metadata
            
        Returns:
            Graph statistics dict
        """
        logger.info(f"Building knowledge graph from {len(chunks)} chunks...")
        
        # Add nodes
        for chunk in chunks:
            node_id = chunk['chunk_id']
            self.graph.add_node(node_id, **chunk)
            self.node_data[node_id] = chunk
        
        # Compute embeddings for all nodes
        logger.info("Computing node embeddings...")
        node_ids = list(self.node_data.keys())
        texts = [self.node_data[nid]['text'] for nid in node_ids]
        
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        for nid, emb in zip(node_ids, embeddings):
            self.node_embeddings[nid] = emb
        
        # Add similarity edges
        logger.info(f"Adding similarity edges (threshold: {self.similarity_threshold})...")
        similarity_edges = self._add_similarity_edges(node_ids)
        
        # Add metadata edges
        logger.info("Adding metadata-based edges...")
        metadata_edges = self._add_metadata_edges()
        
        # Graph statistics
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'similarity_edges': similarity_edges,
            'metadata_edges': metadata_edges,
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'connected_components': nx.number_connected_components(self.graph)
        }
        
        logger.info(f"✓ Knowledge graph built: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        
        return stats
    
    def _add_similarity_edges(self, node_ids: List[str]) -> int:
        """
        Add edges between nodes with high semantic similarity
        
        Strategy:
        - Compute pairwise similarities
        - Add edge if similarity > threshold
        - Avoid O(n²) by using approximate nearest neighbors for large graphs
        """
        edge_count = 0
        n = len(node_ids)
        
        # For small graphs (<1000 nodes), use exact computation
        if n < 1000:
            for i in range(n):
                for j in range(i + 1, n):
                    emb_i = self.node_embeddings[node_ids[i]]
                    emb_j = self.node_embeddings[node_ids[j]]
                    
                    # Cosine similarity
                    similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
                    
                    if similarity >= self.similarity_threshold:
                        self.graph.add_edge(
                            node_ids[i],
                            node_ids[j],
                            weight=float(similarity),
                            edge_type='semantic_similarity'
                        )
                        edge_count += 1
        else:
            # For large graphs, use approximate method (top-k neighbors per node)
            logger.info("Large graph detected, using approximate similarity edges...")
            embeddings_matrix = np.array([self.node_embeddings[nid] for nid in node_ids])
            
            # Compute top-k most similar for each node
            k = 10  # Number of neighbors to consider
            for i, nid in enumerate(node_ids):
                # Compute similarities with all other nodes
                similarities = np.dot(embeddings_matrix, embeddings_matrix[i]) / (
                    np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(embeddings_matrix[i])
                )
                
                # Get top-k (excluding self)
                top_k_indices = np.argsort(similarities)[::-1][1:k+1]
                
                for j in top_k_indices:
                    if similarities[j] >= self.similarity_threshold:
                        # Check if edge already exists
                        if not self.graph.has_edge(nid, node_ids[j]):
                            self.graph.add_edge(
                                nid,
                                node_ids[j],
                                weight=float(similarities[j]),
                                edge_type='semantic_similarity'
                            )
                            edge_count += 1
        
        return edge_count
    
    def _add_metadata_edges(self) -> int:
        """
        Add edges based on metadata connections
        
        Examples:
        - Same well name
        - Same operator
        - Same formation
        - Sequential pages/sections
        """
        edge_count = 0
        
        # Group nodes by metadata
        well_groups = {}
        operator_groups = {}
        formation_groups = {}
        
        for node_id, data in self.node_data.items():
            # Group by well names
            well_names = data.get('well_names', [])
            for well in well_names:
                if well not in well_groups:
                    well_groups[well] = []
                well_groups[well].append(node_id)
            
            # Group by operator (if available)
            operator = data.get('metadata', {}).get('operator')
            if operator:
                if operator not in operator_groups:
                    operator_groups[operator] = []
                operator_groups[operator].append(node_id)
            
            # Group by formation (if available)
            formations = data.get('metadata', {}).get('formations', [])
            for formation in formations:
                if formation not in formation_groups:
                    formation_groups[formation] = []
                formation_groups[formation].append(node_id)
        
        # Add edges within groups
        if 'same_well' in self.metadata_edge_types:
            for well, nodes in well_groups.items():
                if len(nodes) > 1:
                    # Connect all nodes for this well (clique)
                    for i in range(len(nodes)):
                        for j in range(i + 1, len(nodes)):
                            if not self.graph.has_edge(nodes[i], nodes[j]):
                                self.graph.add_edge(
                                    nodes[i],
                                    nodes[j],
                                    weight=1.0,
                                    edge_type='same_well',
                                    well_name=well
                                )
                                edge_count += 1
        
        if 'same_operator' in self.metadata_edge_types:
            for operator, nodes in operator_groups.items():
                if len(nodes) > 1:
                    # Connect nodes with same operator (limited to avoid too many edges)
                    for i in range(min(len(nodes), 50)):
                        for j in range(i + 1, min(len(nodes), 50)):
                            if not self.graph.has_edge(nodes[i], nodes[j]):
                                self.graph.add_edge(
                                    nodes[i],
                                    nodes[j],
                                    weight=0.8,
                                    edge_type='same_operator',
                                    operator=operator
                                )
                                edge_count += 1
        
        return edge_count
    
    def query_graph(self, seed_chunks: List[str], max_hops: int = 2, 
                    max_nodes: int = 20) -> List[Dict]:
        """
        Multi-hop traversal from seed chunks
        
        Args:
            seed_chunks: Starting chunk IDs
            max_hops: Maximum number of hops to traverse
            max_nodes: Maximum number of nodes to return
            
        Returns:
            List of related chunks with hop distances
        """
        if not seed_chunks:
            return []
        
        # BFS traversal from seed chunks
        visited = set()
        queue = [(seed_id, 0) for seed_id in seed_chunks if seed_id in self.graph]
        related_chunks = []
        
        while queue and len(related_chunks) < max_nodes:
            node_id, hop_dist = queue.pop(0)
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            
            # Add to results (skip seeds)
            if hop_dist > 0:
                chunk = self.node_data[node_id].copy()
                chunk['hop_distance'] = hop_dist
                related_chunks.append(chunk)
            
            # Explore neighbors (if within max_hops)
            if hop_dist < max_hops:
                neighbors = self.graph.neighbors(node_id)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, hop_dist + 1))
        
        logger.info(f"Graph traversal: {len(seed_chunks)} seeds → {len(related_chunks)} related chunks")
        
        return related_chunks
    
    def find_shortest_path(self, source_chunk: str, target_chunk: str) -> List[str]:
        """
        Find shortest path between two chunks
        
        Args:
            source_chunk: Source chunk ID
            target_chunk: Target chunk ID
            
        Returns:
            List of chunk IDs along shortest path
        """
        try:
            path = nx.shortest_path(self.graph, source_chunk, target_chunk)
            return path
        except nx.NetworkXNoPath:
            logger.warning(f"No path between {source_chunk} and {target_chunk}")
            return []
    
    def get_node_neighborhood(self, node_id: str, radius: int = 1) -> Set[str]:
        """
        Get all nodes within radius hops of given node
        
        Args:
            node_id: Center node ID
            radius: Neighborhood radius (number of hops)
            
        Returns:
            Set of node IDs in neighborhood
        """
        if node_id not in self.graph:
            return set()
        
        # Use BFS to find nodes within radius
        visited = {node_id}
        queue = [(node_id, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            
            if dist < radius:
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        
        return visited
    
    def detect_communities(self) -> Dict[int, List[str]]:
        """
        Detect document communities using Louvain algorithm
        
        Returns:
            Dict mapping community_id -> list of node IDs
        """
        try:
            communities = nx.community.louvain_communities(self.graph)
            
            community_dict = {}
            for i, community in enumerate(communities):
                community_dict[i] = list(community)
            
            logger.info(f"Detected {len(community_dict)} communities")
            
            return community_dict
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return {}
    
    def get_graph_statistics(self) -> Dict:
        """Get detailed statistics about the knowledge graph"""
        if self.graph.number_of_nodes() == 0:
            return {'status': 'empty'}
        
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph)
        }
        
        # Edge type distribution
        edge_types = {}
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        stats['edge_type_distribution'] = edge_types
        
        # Node degree distribution
        degrees = [d for n, d in self.graph.degree()]
        stats['degree_distribution'] = {
            'min': min(degrees),
            'max': max(degrees),
            'avg': np.mean(degrees),
            'median': np.median(degrees)
        }
        
        return stats
    
    def visualize_subgraph(self, node_ids: List[str], output_path: Optional[str] = None):
        """
        Visualize subgraph (for debugging/analysis)
        
        Args:
            node_ids: List of node IDs to include
            output_path: Optional path to save visualization
        """
        subgraph = self.graph.subgraph(node_ids)
        
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color='lightblue')
            
            # Draw edges (color by type)
            edge_colors = []
            for u, v, data in subgraph.edges(data=True):
                edge_type = data.get('edge_type', 'unknown')
                if edge_type == 'semantic_similarity':
                    edge_colors.append('blue')
                elif edge_type == 'same_well':
                    edge_colors.append('green')
                else:
                    edge_colors.append('gray')
            
            nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, alpha=0.5)
            
            # Draw labels
            labels = {nid: nid.split('_')[-1] for nid in node_ids}  # Simplified labels
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
            
            plt.title("Knowledge Graph Subgraph")
            plt.axis('off')
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved subgraph visualization to {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available for visualization")


def create_knowledge_graph(config: Dict = None) -> KnowledgeGraph:
    """Factory function to create knowledge graph"""
    if config is None:
        config = {}
    
    return KnowledgeGraph(
        embedding_model=config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
        similarity_threshold=config.get('similarity_threshold', 0.7),
        metadata_edge_types=config.get('metadata_edge_types', ['same_well', 'same_operator'])
    )
