"""
Built-in visualization tools for Memlayer.

Provides easy-to-use functions to visualize:
- Knowledge graphs (entities and relationships)
- Memory statistics
- Graph exploration

Works with any Memlayer project - completely agnostic.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Dict, List, Any
import json


class GraphVisualizer:
    """
    Built-in graph visualization for Memlayer.

    Usage:
        from memlayer.visualization import GraphVisualizer

        viz = GraphVisualizer(client.graph_storage)
        viz.show()  # Display in matplotlib
        viz.export_html("graph.html")  # Interactive HTML
        viz.print_summary()  # Text summary
    """

    def __init__(self, graph_storage):
        """
        Initialize visualizer with a graph storage.

        Args:
            graph_storage: NetworkXStorage or MemgraphStorage instance
        """
        self.graph_storage = graph_storage
        self.graph = graph_storage.graph

    def show(
        self,
        figsize: tuple = (16, 10),
        node_size_scale: int = 1000,
        show_edge_labels: bool = True,
        max_label_length: int = 20
    ):
        """
        Display the graph using matplotlib.

        Args:
            figsize: Figure size (width, height)
            node_size_scale: Base size for nodes
            show_edge_labels: Whether to show relationship labels
            max_label_length: Max characters for node labels
        """
        G = self.graph

        if G.number_of_nodes() == 0:
            print("📭 Graph is empty - no entities extracted yet.")
            return

        print(f"📊 Visualizing graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Create figure
        plt.figure(figsize=figsize)

        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Prepare node attributes
        node_colors = []
        node_sizes = []
        node_labels = {}

        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'Concept')
            access_count = G.nodes[node].get('access_count', 1)

            # Color by type
            color_map = {
                "Person": '#FF6B6B',
                "Organization": '#4ECDC4',
                "Project": '#95E1D3',
                "Location": '#F38181',
                "Date": '#FFA07A',
                "Task": '#FFD93D',
                "Concept": '#A8E6CF'
            }
            node_colors.append(color_map.get(node_type, '#A8E6CF'))

            # Size by importance/access
            node_sizes.append(node_size_scale + access_count * 500)

            # Truncate long labels
            label = node if len(node) <= max_label_length else node[:max_label_length] + "..."
            node_labels[node] = label

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='white',
            linewidths=2
        )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#CCCCCC',
            width=2,
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->'
        )

        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=10,
            font_weight='bold',
            font_color='black'
        )

        # Draw edge labels
        if show_edge_labels:
            edge_labels = {}
            for u, v, data in G.edges(data=True):
                edge_type = data.get('type', 'relates_to')
                # Truncate long edge labels
                edge_label = edge_type if len(edge_type) <= 15 else edge_type[:15] + "..."
                edge_labels[(u, v)] = edge_label

            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels,
                font_size=8,
                font_color='#666666'
            )

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B',
                      markersize=10, label='Person'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4',
                      markersize=10, label='Organization'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#95E1D3',
                      markersize=10, label='Project'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD93D',
                      markersize=10, label='Task'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#A8E6CF',
                      markersize=10, label='Concept/Other'),
        ]
        plt.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

        plt.title("Knowledge Graph - Entities and Relationships",
                  fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def print_summary(self, max_items: int = 10):
        """
        Print a text summary of the graph.

        Args:
            max_items: Maximum items to show per section
        """
        G = self.graph

        print("\n" + "=" * 70)
        print("  📊 Knowledge Graph Summary")
        print("=" * 70)

        print(f"\n📈 Statistics:")
        print(f"   Total nodes: {G.number_of_nodes()}")
        print(f"   Total edges: {G.number_of_edges()}")

        # Count by type
        type_counts = {}
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'Unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        print(f"\n📊 Nodes by type:")
        for node_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"   {node_type}: {count}")

        # Most connected nodes
        print(f"\n🔗 Most connected entities:")
        degree = dict(G.degree())
        for node, deg in sorted(degree.items(), key=lambda x: -x[1])[:max_items]:
            node_type = G.nodes[node].get('type', 'Concept')
            print(f"   {node} ({node_type}): {deg} connections")

        # Sample entities
        print(f"\n📝 Sample entities:")
        for node in list(G.nodes())[:max_items]:
            attrs = G.nodes[node]
            node_type = attrs.get('type', 'Concept')
            access_count = attrs.get('access_count', 0)
            print(f"   • {node} ({node_type}) - accessed {access_count} times")

        if G.number_of_nodes() > max_items:
            print(f"   ... and {G.number_of_nodes() - max_items} more")

        # Sample relationships
        print(f"\n🔗 Sample relationships:")
        for u, v, data in list(G.edges(data=True))[:max_items]:
            edge_type = data.get('type', 'relates_to')
            print(f"   • {u} --[{edge_type}]--> {v}")

        if G.number_of_edges() > max_items:
            print(f"   ... and {G.number_of_edges() - max_items} more")

        print("\n" + "=" * 70)

    def export_json(self) -> Dict[str, Any]:
        """
        Export graph as JSON data structure.

        Returns:
            Dict with 'nodes' and 'edges' keys
        """
        G = self.graph

        graph_data = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "node_count": G.number_of_nodes(),
                "edge_count": G.number_of_edges()
            }
        }

        # Export nodes
        for node in G.nodes():
            attrs = G.nodes[node]
            graph_data["nodes"].append({
                "id": node,
                "type": attrs.get("type", "Concept"),
                "access_count": attrs.get("access_count", 0),
                "importance_score": attrs.get("importance_score", 0.5),
                "status": attrs.get("status", "active")
            })

        # Export edges
        for u, v, data in G.edges(data=True):
            graph_data["edges"].append({
                "source": u,
                "target": v,
                "type": data.get("type", "relates_to")
            })

        return graph_data

    def export_html(self, filename: str = "graph.html"):
        """
        Export an interactive HTML visualization using vis.js.

        Args:
            filename: Output filename
        """
        graph_data = self.export_json()

        # Convert to vis.js format
        vis_nodes = []
        for node in graph_data["nodes"]:
            # Color by type
            color_map = {
                "Person": '#FF6B6B',
                "Organization": '#4ECDC4',
                "Project": '#95E1D3',
                "Location": '#F38181',
                "Date": '#FFA07A',
                "Task": '#FFD93D',
                "Concept": '#A8E6CF'
            }

            vis_nodes.append({
                "id": node["id"],
                "label": node["id"],
                "title": f"{node['type']} (accessed {node['access_count']} times)",
                "color": color_map.get(node["type"], '#A8E6CF'),
                "value": node["access_count"] + 1
            })

        vis_edges = []
        for edge in graph_data["edges"]:
            vis_edges.append({
                "from": edge["source"],
                "to": edge["target"],
                "label": edge["type"],
                "arrows": "to"
            })

        # HTML template
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Memlayer Knowledge Graph</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        #mynetwork {{
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .info {{
            background: white;
            padding: 15px;
            margin-top: 20px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <h1>📊 Memlayer Knowledge Graph</h1>
    <div id="mynetwork"></div>
    <div class="info">
        <h3>Graph Statistics</h3>
        <p><strong>Nodes:</strong> {graph_data['metadata']['node_count']}</p>
        <p><strong>Edges:</strong> {graph_data['metadata']['edge_count']}</p>
        <p><strong>Legend:</strong></p>
        <ul>
            <li><span style="color: #FF6B6B;">●</span> Person</li>
            <li><span style="color: #4ECDC4;">●</span> Organization</li>
            <li><span style="color: #95E1D3;">●</span> Project</li>
            <li><span style="color: #FFD93D;">●</span> Task</li>
            <li><span style="color: #A8E6CF;">●</span> Concept/Other</li>
        </ul>
    </div>

    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(vis_nodes)});
        var edges = new vis.DataSet({json.dumps(vis_edges)});

        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            nodes: {{
                shape: 'dot',
                scaling: {{
                    min: 10,
                    max: 30
                }},
                font: {{
                    size: 14,
                    face: 'Arial'
                }}
            }},
            edges: {{
                arrows: {{
                    to: {{enabled: true, scaleFactor: 0.5}}
                }},
                color: {{color: '#848484'}},
                font: {{
                    size: 10,
                    align: 'middle'
                }},
                smooth: {{
                    type: 'continuous'
                }}
            }},
            physics: {{
                stabilization: false,
                barnesHut: {{
                    gravitationalConstant: -8000,
                    springConstant: 0.001,
                    springLength: 200
                }}
            }}
        }};
        var network = new vis.Network(container, data, options);

        // Click handler
        network.on("click", function (params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                alert("Node: " + node.label + "\\nType: " + node.title);
            }}
        }});
    </script>
</body>
</html>
"""

        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Interactive graph exported to: {filename}")
        print(f"   Open in browser to explore!")
