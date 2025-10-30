import ast
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import hashlib


class CFGNode:
    """Represents a node in the Control Flow Graph"""
    
    def __init__(self, node_id: int, node_type: str, code: str, lineno: int = -1):
        self.id = node_id
        self.type = node_type
        self.code = code
        self.lineno = lineno
        self.successors = []
        self.predecessors = []
        
    def add_successor(self, node):
        if node not in self.successors:
            self.successors.append(node)
            node.predecessors.append(self)
    
    def __repr__(self):
        return f"CFGNode({self.id}, {self.type}, {self.code[:30]}...)"


class CFGExtractor(ast.NodeVisitor):
    """Extracts Control Flow Graph from Python AST"""
    
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_counter = 0
        self.current_node = None
        self.entry_node = None
        self.exit_node = None
        self.loop_stack = []  # Stack for handling loops
        self.after_loop_stack = []  # Stack for nodes after loops
        
    def create_node(self, node_type: str, code: str, lineno: int = -1) -> CFGNode:
        """Create a new CFG node"""
        cfg_node = CFGNode(self.node_counter, node_type, code, lineno)
        self.nodes.append(cfg_node)
        self.node_counter += 1
        return cfg_node
    
    def extract_cfg(self, code: str) -> Tuple[List[CFGNode], List[Tuple[int, int]]]:
        """Extract CFG from Python code string"""
        tree = ast.parse(code)
        
        # Create entry and exit nodes
        self.entry_node = self.create_node("ENTRY", "START", -1)
        self.exit_node = self.create_node("EXIT", "END", -1)
        self.current_node = self.entry_node
        
        # Visit the AST
        self.visit(tree)
        
        # Connect last node to exit
        if self.current_node and self.current_node != self.exit_node:
            self.current_node.add_successor(self.exit_node)
        
        # Extract edges from nodes
        for node in self.nodes:
            for successor in node.successors:
                self.edges.append((node.id, successor.id))
        
        return self.nodes, self.edges
    
    def visit_Module(self, node):
        """Visit module node"""
        for stmt in node.body:
            self.visit(stmt)
    
    def visit_FunctionDef(self, node):
        """Visit function definition"""
        func_code = f"def {node.name}(...)"
        func_node = self.create_node("FUNCTION_DEF", func_code, node.lineno)
        
        if self.current_node:
            self.current_node.add_successor(func_node)
        
        self.current_node = func_node
        
        # Visit function body
        for stmt in node.body:
            self.visit(stmt)
    
    def visit_If(self, node):
        """Visit if statement"""
        # Create condition node
        cond_code = ast.unparse(node.test) if hasattr(ast, 'unparse') else "if_condition"
        cond_node = self.create_node("IF_CONDITION", f"if {cond_code}", node.lineno)
        
        if self.current_node:
            self.current_node.add_successor(cond_node)
        
        # Create branch for then body
        then_entry = self.create_node("THEN_BRANCH", "then", node.body[0].lineno if node.body else -1)
        cond_node.add_successor(then_entry)
        
        self.current_node = then_entry
        for stmt in node.body:
            self.visit(stmt)
        then_exit = self.current_node
        
        # Create branch for else body
        if node.orelse:
            else_entry = self.create_node("ELSE_BRANCH", "else", node.orelse[0].lineno if node.orelse else -1)
            cond_node.add_successor(else_entry)
            
            self.current_node = else_entry
            for stmt in node.orelse:
                self.visit(stmt)
            else_exit = self.current_node
        else:
            else_exit = cond_node
        
        # Create merge node
        merge_node = self.create_node("MERGE", "endif", -1)
        if then_exit:
            then_exit.add_successor(merge_node)
        if else_exit and else_exit != cond_node:
            else_exit.add_successor(merge_node)
        elif not node.orelse:
            cond_node.add_successor(merge_node)
        
        self.current_node = merge_node
    
    def visit_While(self, node):
        """Visit while loop"""
        # Create loop condition node
        cond_code = ast.unparse(node.test) if hasattr(ast, 'unparse') else "while_condition"
        loop_node = self.create_node("WHILE_CONDITION", f"while {cond_code}", node.lineno)
        
        if self.current_node:
            self.current_node.add_successor(loop_node)
        
        # Create loop body entry
        body_entry = self.create_node("LOOP_BODY", "loop_body", node.body[0].lineno if node.body else -1)
        loop_node.add_successor(body_entry)
        
        # Push loop info onto stack for break/continue
        self.loop_stack.append(loop_node)
        after_loop = self.create_node("AFTER_LOOP", "after_loop", -1)
        self.after_loop_stack.append(after_loop)
        
        # Visit loop body
        self.current_node = body_entry
        for stmt in node.body:
            self.visit(stmt)
        
        # Connect back to loop condition
        if self.current_node:
            self.current_node.add_successor(loop_node)
        
        # Connect loop exit
        loop_node.add_successor(after_loop)
        
        self.loop_stack.pop()
        self.after_loop_stack.pop()
        self.current_node = after_loop
    
    def visit_For(self, node):
        """Visit for loop"""
        # Create loop node
        iter_code = ast.unparse(node.iter) if hasattr(ast, 'unparse') else "iterable"
        target_code = ast.unparse(node.target) if hasattr(ast, 'unparse') else "var"
        loop_node = self.create_node("FOR_LOOP", f"for {target_code} in {iter_code}", node.lineno)
        
        if self.current_node:
            self.current_node.add_successor(loop_node)
        
        # Create loop body entry
        body_entry = self.create_node("LOOP_BODY", "loop_body", node.body[0].lineno if node.body else -1)
        loop_node.add_successor(body_entry)
        
        # Push loop info onto stack
        self.loop_stack.append(loop_node)
        after_loop = self.create_node("AFTER_LOOP", "after_loop", -1)
        self.after_loop_stack.append(after_loop)
        
        # Visit loop body
        self.current_node = body_entry
        for stmt in node.body:
            self.visit(stmt)
        
        # Connect back to loop
        if self.current_node:
            self.current_node.add_successor(loop_node)
        
        # Connect loop exit
        loop_node.add_successor(after_loop)
        
        self.loop_stack.pop()
        self.after_loop_stack.pop()
        self.current_node = after_loop
    
    def visit_Break(self, node):
        """Visit break statement"""
        break_node = self.create_node("BREAK", "break", node.lineno)
        if self.current_node:
            self.current_node.add_successor(break_node)
        
        # Connect to after loop node
        if self.after_loop_stack:
            break_node.add_successor(self.after_loop_stack[-1])
        
        self.current_node = None  # No successor in current path
    
    def visit_Continue(self, node):
        """Visit continue statement"""
        continue_node = self.create_node("CONTINUE", "continue", node.lineno)
        if self.current_node:
            self.current_node.add_successor(continue_node)
        
        # Connect back to loop condition
        if self.loop_stack:
            continue_node.add_successor(self.loop_stack[-1])
        
        self.current_node = None  # No successor in current path
    
    def visit_Return(self, node):
        """Visit return statement"""
        ret_code = ast.unparse(node.value) if node.value and hasattr(ast, 'unparse') else "None"
        return_node = self.create_node("RETURN", f"return {ret_code}", node.lineno)
        
        if self.current_node:
            self.current_node.add_successor(return_node)
        
        # Connect to exit node
        return_node.add_successor(self.exit_node)
        self.current_node = None  # No successor after return
    
    def visit_Assign(self, node):
        """Visit assignment statement"""
        assign_code = ast.unparse(node) if hasattr(ast, 'unparse') else "assignment"
        assign_node = self.create_node("ASSIGN", assign_code, node.lineno)
        
        if self.current_node:
            self.current_node.add_successor(assign_node)
        
        self.current_node = assign_node
    
    def visit_Expr(self, node):
        """Visit expression statement"""
        expr_code = ast.unparse(node) if hasattr(ast, 'unparse') else "expression"
        expr_node = self.create_node("EXPR", expr_code, node.lineno)
        
        if self.current_node:
            self.current_node.add_successor(expr_node)
        
        self.current_node = expr_node
    
    def generic_visit(self, node):
        """Handle other node types"""
        if isinstance(node, ast.stmt):
            stmt_code = ast.unparse(node) if hasattr(ast, 'unparse') else str(type(node).__name__)
            stmt_node = self.create_node(type(node).__name__.upper(), stmt_code, getattr(node, 'lineno', -1))
            
            if self.current_node:
                self.current_node.add_successor(stmt_node)
            
            self.current_node = stmt_node
        
        super().generic_visit(node)


def create_node_features(nodes: List[CFGNode], feature_dim: int = 128) -> torch.Tensor:
    """Create node feature vectors for PyG"""
    features = []
    
    for node in nodes:
        # Create feature vector based on node properties
        feature = torch.zeros(feature_dim)
        
        # Encode node type (first 20 dimensions)
        node_types = ["ENTRY", "EXIT", "FUNCTION_DEF", "IF_CONDITION", "THEN_BRANCH", 
                     "ELSE_BRANCH", "MERGE", "WHILE_CONDITION", "FOR_LOOP", "LOOP_BODY",
                     "AFTER_LOOP", "BREAK", "CONTINUE", "RETURN", "ASSIGN", "EXPR"]
        
        if node.type in node_types:
            type_idx = node_types.index(node.type)
            feature[type_idx] = 1.0
        
        # Encode code hash (next 32 dimensions)
        code_hash = hashlib.md5(node.code.encode()).hexdigest()
        for i, char in enumerate(code_hash[:32]):
            feature[20 + i] = ord(char) / 255.0
        
        # Add line number information (normalized)
        if node.lineno > 0:
            feature[52] = min(node.lineno / 1000.0, 1.0)
        
        # Add node degree information
        feature[53] = len(node.successors) / 10.0
        feature[54] = len(node.predecessors) / 10.0
        
        features.append(feature)
    
    return torch.stack(features)


def cfg_to_pyg_data(code: str, label: Optional[int] = None) -> Data:
    """
    Convert Python code to PyG Data object with CFG structure
    
    Args:
        code: Python source code string
        label: Optional label for the graph (for classification tasks)
    
    Returns:
        PyG Data object containing the CFG
    """
    # Extract CFG
    extractor = CFGExtractor()
    nodes, edges = extractor.extract_cfg(code)
    
    # Create node features
    x = create_node_features(nodes)
    
    # Create edge index
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index)
    
    # Add optional label
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    
    # Add metadata
    data.num_nodes = len(nodes)
    data.code = code
    
    return data


def visualize_cfg(data: Data, save_path: Optional[str] = None):
    """Visualize the CFG using networkx and matplotlib"""
    import matplotlib.pyplot as plt
    
    # Create networkx graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(data.num_nodes):
        G.add_node(i)
    
    # Add edges
    edges = data.edge_index.t().numpy()
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    # Draw graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, arrows=True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example Python code
    x = ("import os\n\ndef load_image(filename):\n    try:\n        with open(os.path.join('assets', filename), 'rb') as f:\n            image_data = f.read()\n        return image_data\n    except IOError:\n        print(f\"Error loading image {filename}\")\n        return None\n\nimage_data = load_image('player.png')\nif image_data is not None:\n    print(\"Image loaded successfully\")")

    
    # Extract CFG and convert to PyG data
    pyg_data = cfg_to_pyg_data(x, label=0)
    
    print(f"Number of nodes: {pyg_data.num_nodes}")
    print(f"Number of edges: {pyg_data.edge_index.shape[1]}")
    print(f"Node feature shape: {pyg_data.x.shape}")
    print(f"Edge index shape: {pyg_data.edge_index.shape}")
    
    # Print node information
    extractor = CFGExtractor()
    nodes, _ = extractor.extract_cfg(x)
    print("\nCFG Nodes:")
    for node in nodes[:10]:  # Print first 10 nodes
        print(f"  Node {node.id}: {node.type} - {node.code[:50]}")