import ast
import torch
from torch_geometric.data import Data
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os, json
# from cfg import cfg_to_pyg_data
# Config
MODEL_PATH = "/home/xuhaoche/.llama/checkpoints/Llama3.1-8B-Instruct"  
OUT_DIR = "processed_data/validation_data"
MAX_LEN = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only = True)
tokenizer.pad_token = tokenizer.eos_token

os.makedirs(OUT_DIR, exist_ok=True)

# AST 
class ASTGraphBuilder(ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_idx = 0
        self.stack = []

    def visit(self, node):
        current_id = self.node_idx
        self.nodes.append(type(node).__name__)
        self.node_idx += 1

        if self.stack:
            parent_id = self.stack[-1]
            self.edges.append((parent_id, current_id))

        self.stack.append(current_id)
        super().visit(node)
        self.stack.pop()

    def build(self, code):
        self.__init__()
        try:
            tree = ast.parse(code)
            self.visit(tree)
            return self.nodes, self.edges
        except Exception as e:
            return [], []

# # === Convert to feature vector ===
# def get_node_features(node_types):
#     unique_types = list(set(node_types))
#     type2id = {typ: i for i, typ in enumerate(unique_types)}
#     features = [torch.nn.functional.one_hot(torch.tensor(type2id[typ]), num_classes=len(type2id)).float() for typ in node_types]
#     return torch.stack(features)


#CFG
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
    

    return data

class DFGBuilder(ast.NodeVisitor):
    def __init__(self):
        self.defs = {}  # var name -> last definition node id
        self.edges = []
        self.nodes = []
        self.node_idx = 0

    def visit(self, node):
        current_id = self.node_idx
        self.nodes.append(type(node).__name__)
        self.node_idx += 1

        # Variable def/use detection - rough example
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                self.defs[node.id] = current_id
            elif isinstance(node.ctx, ast.Load):
                if node.id in self.defs:
                    self.edges.append((self.defs[node.id], current_id))

        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def build(self, code):
        self.__init__()
        try:
            tree = ast.parse(code)
            self.visit(tree)
            return self.nodes, self.edges
        except Exception:
            return [], []
        

def nx_to_pyg(g):
    if len(g.nodes) == 0:
        return Data(x=torch.empty((0, 1)), edge_index=torch.empty((2, 0), dtype=torch.long))

    # Map original node IDs to contiguous indices [0 .. N-1]
    mapping = {node: i for i, node in enumerate(g.nodes)}
    edges = [(mapping[u], mapping[v]) for u, v in g.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    # Get node labels in order of nodes in graph (should be 10)
    labels = [g.nodes[n].get("label", "None") for n in g.nodes]
    unique_labels = sorted(set(labels))  # Sort for consistent ordering
    label2id = {l: i for i, l in enumerate(unique_labels)}

    # Convert each node label to one-hot vector with dimension = number of unique labels in this graph (should be small, like 5-10)
    features = [torch.nn.functional.one_hot(torch.tensor(label2id[l]), num_classes=len(unique_labels)).float() for l in labels]
    x = torch.stack(features) if features else torch.empty((len(labels), 1))

    return Data(x=x, edge_index=edge_index)


def get_ast_node_features_global(node_types):
    indices = []
    for typ in node_types:
        if typ in type2id:
            indices.append(type2id[typ])
        else:
            print(f"[Warning] Unknown node type: {typ}")  # Debug print
            indices.append(0)  # or map to some default 'unknown' class id
    features = torch.nn.functional.one_hot(torch.tensor(indices), num_classes=num_classes).float()
    target_dim = 128
    if features.shape[1] < target_dim:
        pad_size = target_dim - features.shape[1]
        pad_tensor = torch.zeros(features.shape[0], pad_size)
        features = torch.cat([features, pad_tensor], dim=1)
    elif features.shape[1] > target_dim:
        features = features[:, :target_dim]  # truncate
    return features

def get_dfg_node_features_global(node_types):
    indices = [dfg_type2id.get(typ, 0) for typ in node_types]
    features = torch.nn.functional.one_hot(torch.tensor(indices), num_classes=dfg_num_classes).float()
    target_dim = 128
    if features.shape[1] < target_dim:
        pad_size = target_dim - features.shape[1]
        pad_tensor = torch.zeros(features.shape[0], pad_size)
        features = torch.cat([features, pad_tensor], dim=1)
    elif features.shape[1] > target_dim:
        features = features[:, :target_dim]  # truncate
    return features

def print_validation_summary(validation_results):
    """Print a summary of graph validation results"""
    print(f"AST: {'✓' if validation_results['ast_valid'] else '✗'} "
          f"({validation_results['ast_nodes']} nodes, {validation_results['ast_edges']} edges)")
    print(f"CFG: {'✓' if validation_results['cfg_valid'] else '✗'} "
          f"({validation_results['cfg_nodes']} nodes, {validation_results['cfg_edges']} edges)")
    print(f"DFG: {'✓' if validation_results['dfg_valid'] else '✗'} "
          f"({validation_results['dfg_nodes']} nodes, {validation_results['dfg_edges']} edges)")
    print(f"All graphs valid: {'✓' if validation_results['all_valid'] else '✗'}")
    print("-" * 50)


def main():
# Load Dataset
    ds = load_dataset("/home/xuhaoche/GACO/preprocessing/InstructCoder", split="validation")

    global_node_types = set()
    for ex in ds:
        node_types, _ = ASTGraphBuilder().build(ex["input"])
        global_node_types.update(node_types)

    global_node_types = sorted(global_node_types)
    type2id = {typ: i for i, typ in enumerate(global_node_types)}
    num_classes = len(global_node_types)

    print(f"Collected {num_classes} unique AST node types globally.")


    global_dfg_node_types = set()
    for ex in ds:
        dfg_builder = DFGBuilder()
        node_types, _ = dfg_builder.build(ex["input"])
        global_dfg_node_types.update(node_types)

    global_dfg_node_types = sorted(global_dfg_node_types)
    dfg_type2id = {typ: i for i, typ in enumerate(global_dfg_node_types)}
    dfg_num_classes = len(global_dfg_node_types)

    print(f"Collected {dfg_num_classes} unique DFG node types globally.")


    # Process
    processed = []

    def save_feature_mappings(output_dir):
        feature_mappings = {
            'ast_type2id': type2id,
            'ast_num_classes': num_classes,
            'dfg_type2id': dfg_type2id,
            'dfg_num_classes': dfg_num_classes,
            'target_dim': 128,
        }
        torch.save(feature_mappings, os.path.join(output_dir, 'feature_mappings.pt'))


    print("Processing samples with AST, CFG, DFG graphs...")


    total_samples = 0
    valid_samples = 0
    ast_failures = 0
    cfg_failures = 0 
    dfg_failures = 0
    
    for i, ex in tqdm(enumerate(ds), total=len(ds)):
        try:
            prompt = f"### Instruction:\n{ex['instruction']}\n\n### Input Code:\n{ex['input']}\n\n### Edited Code:"
            output = ex['output']

            tok = tokenizer(prompt, truncation=True, padding="max_length", max_length=MAX_LEN)
            labels = tokenizer(output, truncation=True, padding="max_length", max_length=MAX_LEN)

            # AST
            ast_builder = ASTGraphBuilder()
            node_types, edge_list = ast_builder.build(ex["input"])
            if len(node_types) == 0:
                continue
            x_ast = get_ast_node_features_global(node_types)
            edge_index_ast = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
            graph_ast = Data(x=x_ast, edge_index=edge_index_ast)

            # CFG
            graph_cfg = cfg_to_pyg_data(ex["input"], label=0)

            # DFG
            dfg_builder = DFGBuilder()
            dfg_nodes, dfg_edges = dfg_builder.build(ex["input"])
            
            if len(dfg_nodes) > 0:
                x_dfg = get_dfg_node_features_global(dfg_nodes)
            else:
                x_dfg = torch.empty((0, dfg_num_classes))

            edge_index_dfg = torch.tensor(dfg_edges, dtype=torch.long).t().contiguous() if dfg_edges else torch.empty((2, 0), dtype=torch.long)
            graph_dfg = Data(x=x_dfg, edge_index=edge_index_dfg)

            # Validate graphs
            validation_results = validate_graphs(graph_ast, graph_cfg, graph_dfg)
            
            total_samples += 1
            
            # Print validation for first few samples or failed samples
            if i < 5 or not validation_results['all_valid']:
                print(f"\nSample {i}:")
                print_validation_summary(validation_results)
            
            # Track statistics
            if validation_results['all_valid']:
                valid_samples += 1
            else:
                if not validation_results['ast_valid']:
                    ast_failures += 1
                if not validation_results['cfg_valid']:
                    cfg_failures += 1
                if not validation_results['dfg_valid']:
                    dfg_failures += 1

            # Only add to processed if all graphs are valid (optional - you might want to keep partial data)
            # if validation_results['all_valid']:
            processed.append({
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "labels": labels["input_ids"],
                "graph_ast": graph_ast,
                "graph_cfg": graph_cfg,
                "graph_dfg": graph_dfg,
                "validation": validation_results  # Include validation info
            })

        except Exception as e:
            print(f"[Error] Example {i}: {e}")
            continue

    # Print final statistics
    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples processed: {total_samples}")
    print(f"Samples with all 3 graphs: {valid_samples} ({valid_samples/total_samples*100:.1f}%)")
    print(f"AST failures: {ast_failures}")
    print(f"CFG failures: {cfg_failures}")
    print(f"DFG failures: {dfg_failures}")
    print(f"Valid samples added to dataset: {len(processed)}")
    
    # return processed
    # for i, ex in tqdm(enumerate(ds), total=len(ds)):
    #     try:
    #         prompt = f"### Instruction:\n{ex['instruction']}\n\n### Input Code:\n{ex['input']}\n\n### Edited Code:"
    #         output = ex['output']

    #         tok = tokenizer(prompt, truncation=True, padding="max_length", max_length=MAX_LEN)
    #         labels = tokenizer(output, truncation=True, padding="max_length", max_length=MAX_LEN)

    #         # AST
    #         ast_builder = ASTGraphBuilder()
    #         node_types, edge_list = ast_builder.build(ex["input"])
    #         if len(node_types) == 0:
    #             continue
    #         # x_ast = torch.nn.functional.one_hot(torch.arange(len(node_types)), num_classes=len(set(node_types))).float()  # simpler feature, or use get_node_features(node_types)
    #         x_ast = get_ast_node_features_global(node_types)
    #         edge_index_ast = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
    #         graph_ast = Data(x=x_ast, edge_index=edge_index_ast)

    #         # CFG
    #         graph_cfg = cfg_to_pyg_data(ex["input"], label=0)
            

    #         # DFG
    #         dfg_builder = DFGBuilder()
    #         dfg_nodes, dfg_edges = dfg_builder.build(ex["input"])
    #         # graph_dfg = Data(
    #         #     x=torch.nn.functional.one_hot(torch.arange(len(dfg_nodes)), num_classes=len(set(dfg_nodes))).float() if len(dfg_nodes) > 0 else torch.empty((0, 1)),
    #         #     edge_index=torch.tensor(dfg_edges, dtype=torch.long).t().contiguous() if dfg_edges else torch.empty((2, 0), dtype=torch.long),
    #         # )cccccbrgddingjibjhbntlhhuffbdlkjvbfulvlhceuk
            

    #         if len(dfg_nodes) > 0:
    #             x_dfg = get_dfg_node_features_global(dfg_nodes)
    #         else:
    #             x_dfg = torch.empty((0, dfg_num_classes))

    #         edge_index_dfg = torch.tensor(dfg_edges, dtype=torch.long).t().contiguous() if dfg_edges else torch.empty((2, 0), dtype=torch.long)

    #         graph_dfg = Data(x=x_dfg, edge_index=edge_index_dfg)

    #         processed.append({
    #             "input_ids": tok["input_ids"],
    #             "attention_mask": tok["attention_mask"],
    #             "labels": labels["input_ids"],
    #             "graph_ast": graph_ast,
    #             "graph_cfg": graph_cfg,
    #             "graph_dfg": graph_dfg,
    #         })

     

    #     except Exception as e:
    #         print(f"[Error] Example {i}: {e}")
    #         continue


    save_feature_mappings(OUT_DIR)
    print("Done preprocessing all samples!")

if __name__ == "__main__":
    main()
