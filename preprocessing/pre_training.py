import ast
import torch
from torch_geometric.data import Data
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import hashlib
from typing import List, Tuple, Optional
import networkx as nx

# Configuration
MODEL_PATH = "/home/xuhaoche/.llama/checkpoints/Llama3.1-8B-Instruct"  
OUT_DIR = "processed_data/training_data"
MAX_LEN = 512
GRAPH_TOKEN_RESERVE = 128  # Reserved tokens for graph
EFFECTIVE_MAX_LEN = MAX_LEN - GRAPH_TOKEN_RESERVE  # 384 tokens for text

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
os.makedirs(OUT_DIR, exist_ok=True)

class ASTGraphBuilder(ast.NodeVisitor):
    """Extract Abstract Syntax Tree structure"""
    
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
        except Exception:
            return [], []

class DFGBuilder(ast.NodeVisitor):
    """Extract Data Flow Graph - tracks variable definitions and uses"""
    
    def __init__(self):
        self.var_defs = {}  # var_name -> list of definition node_ids
        self.var_uses = {}  # var_name -> list of use node_ids
        self.nodes = []     # list of (node_id, node_type, var_name_if_applicable)
        self.edges = []     # list of (def_node_id, use_node_id)
        self.node_idx = 0

    def create_node(self, node_type, var_name=None):
        """Create a new DFG node"""
        node_id = self.node_idx
        self.nodes.append((node_id, node_type, var_name))
        self.node_idx += 1
        return node_id

    def visit_Name(self, node):
        """Handle variable names (the core of DFG)"""
        var_name = node.id
        
        if isinstance(node.ctx, ast.Store):
            # Variable definition
            def_node_id = self.create_node("VAR_DEF", var_name)
            if var_name not in self.var_defs:
                self.var_defs[var_name] = []
            self.var_defs[var_name].append(def_node_id)
            
        elif isinstance(node.ctx, ast.Load):
            # Variable use
            use_node_id = self.create_node("VAR_USE", var_name)
            if var_name not in self.var_uses:
                self.var_uses[var_name] = []
            self.var_uses[var_name].append(use_node_id)
            
            # Connect to most recent definition
            if var_name in self.var_defs and self.var_defs[var_name]:
                latest_def = self.var_defs[var_name][-1]
                self.edges.append((latest_def, use_node_id))

    def visit_FunctionDef(self, node):
        """Handle function definitions"""
        func_node_id = self.create_node("FUNC_DEF", node.name)
        
        # Function name is also a definition
        if node.name not in self.var_defs:
            self.var_defs[node.name] = []
        self.var_defs[node.name].append(func_node_id)
        
        # Visit function body
        for stmt in node.body:
            self.visit(stmt)

    def visit_Assign(self, node):
        """Handle assignments"""
        assign_node_id = self.create_node("ASSIGN")
        
        # Visit right side first (uses)
        self.visit(node.value)
        
        # Then left side (definitions)
        for target in node.targets:
            self.visit(target)

    def generic_visit(self, node):
        """Visit other node types"""
        super().generic_visit(node)

    def build(self, code):
        """Build DFG from code string"""
        self.__init__()
        try:
            tree = ast.parse(code)
            self.visit(tree)
            
            # Return node types and edges
            node_types = [node_type for _, node_type, _ in self.nodes]
            return node_types, self.edges
        except Exception:
            return [], []

class CFGNode:
    """Control Flow Graph node"""
    
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

class CFGExtractor(ast.NodeVisitor):
    """Extract Control Flow Graph"""
    
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_counter = 0
        self.current_node = None
        self.entry_node = None
        self.exit_node = None
        self.loop_stack = []
        self.after_loop_stack = []
        
    def create_node(self, node_type: str, code: str, lineno: int = -1) -> CFGNode:
        cfg_node = CFGNode(self.node_counter, node_type, code, lineno)
        self.nodes.append(cfg_node)
        self.node_counter += 1
        return cfg_node
    
    def extract_cfg(self, code: str) -> Tuple[List[CFGNode], List[Tuple[int, int]]]:
        tree = ast.parse(code)
        
        self.entry_node = self.create_node("ENTRY", "START", -1)
        self.exit_node = self.create_node("EXIT", "END", -1)
        self.current_node = self.entry_node
        
        self.visit(tree)
        
        if self.current_node and self.current_node != self.exit_node:
            self.current_node.add_successor(self.exit_node)
        
        for node in self.nodes:
            for successor in node.successors:
                self.edges.append((node.id, successor.id))
        
        return self.nodes, self.edges
    
    def visit_Module(self, node):
        for stmt in node.body:
            self.visit(stmt)
    
    def visit_FunctionDef(self, node):
        func_code = f"def {node.name}(...)"
        func_node = self.create_node("FUNCTION_DEF", func_code, node.lineno)
        
        if self.current_node:
            self.current_node.add_successor(func_node)
        
        self.current_node = func_node
        
        for stmt in node.body:
            self.visit(stmt)
    
    def visit_If(self, node):
        cond_code = ast.unparse(node.test) if hasattr(ast, 'unparse') else "if_condition"
        cond_node = self.create_node("IF_CONDITION", f"if {cond_code}", node.lineno)
        
        if self.current_node:
            self.current_node.add_successor(cond_node)
        
        then_entry = self.create_node("THEN_BRANCH", "then", node.body[0].lineno if node.body else -1)
        cond_node.add_successor(then_entry)
        
        self.current_node = then_entry
        for stmt in node.body:
            self.visit(stmt)
        then_exit = self.current_node
        
        if node.orelse:
            else_entry = self.create_node("ELSE_BRANCH", "else", node.orelse[0].lineno if node.orelse else -1)
            cond_node.add_successor(else_entry)
            
            self.current_node = else_entry
            for stmt in node.orelse:
                self.visit(stmt)
            else_exit = self.current_node
        else:
            else_exit = cond_node
        
        merge_node = self.create_node("MERGE", "endif", -1)
        if then_exit:
            then_exit.add_successor(merge_node)
        if else_exit and else_exit != cond_node:
            else_exit.add_successor(merge_node)
        elif not node.orelse:
            cond_node.add_successor(merge_node)
        
        self.current_node = merge_node
    
    def visit_Assign(self, node):
        assign_code = ast.unparse(node) if hasattr(ast, 'unparse') else "assignment"
        assign_node = self.create_node("ASSIGN", assign_code, node.lineno)
        
        if self.current_node:
            self.current_node.add_successor(assign_node)
        
        self.current_node = assign_node
    
    def generic_visit(self, node):
        if isinstance(node, ast.stmt):
            stmt_code = ast.unparse(node) if hasattr(ast, 'unparse') else str(type(node).__name__)
            stmt_node = self.create_node(type(node).__name__.upper(), stmt_code, getattr(node, 'lineno', -1))
            
            if self.current_node:
                self.current_node.add_successor(stmt_node)
            
            self.current_node = stmt_node
        
        super().generic_visit(node)

def create_cfg_features(nodes: List[CFGNode], feature_dim: int = 128) -> torch.Tensor:
    """Create node features for CFG"""
    features = []
    
    # Expanded node types for better CFG representation
    node_types = [
        "ENTRY", "EXIT", "FUNCTION_DEF", "FUNC_EXIT", 
        "IF_CONDITION", "THEN_BRANCH", "ELSE_BRANCH", "MERGE",
        "WHILE_HEADER", "WHILE_BODY", "AFTER_WHILE",
        "FOR_HEADER", "FOR_BODY", "AFTER_FOR", "FOR_ELSE", "FOR_MERGE",
        "TRY", "EXCEPT", "TRY_ELSE", "FINALLY", "TRY_MERGE",
        "BREAK", "CONTINUE", "RETURN", "ASSIGN", "EXPR"
    ]
    
    for node in nodes:
        feature = torch.zeros(feature_dim)
        
        # One-hot encoding for node type (first 25 dimensions)
        if node.type in node_types:
            type_idx = node_types.index(node.type)
            feature[type_idx] = 1.0
        
        # Code content hash (dimensions 25-56)
        code_hash = hashlib.md5(node.code.encode()).hexdigest()
        for i, char in enumerate(code_hash[:32]):
            feature[25 + i] = ord(char) / 255.0
        
        # Line number information (dimension 57)
        if node.lineno > 0:
            feature[57] = min(node.lineno / 1000.0, 1.0)
        
        # Control flow properties (dimensions 58-62)
        feature[58] = len(node.successors) / 10.0    # Out-degree
        feature[59] = len(node.predecessors) / 10.0  # In-degree
        
        # Control flow type indicators (dimensions 60-65)
        if "LOOP" in node.type or "WHILE" in node.type or "FOR" in node.type:
            feature[60] = 1.0  # Is loop-related
        if "IF" in node.type or "CONDITION" in node.type:
            feature[61] = 1.0  # Is conditional
        if "BRANCH" in node.type:
            feature[62] = 1.0  # Is branch
        if node.type in ["BREAK", "CONTINUE", "RETURN"]:
            feature[63] = 1.0  # Is control transfer
        if "TRY" in node.type or "EXCEPT" in node.type:
            feature[64] = 1.0  # Is exception handling
        
        features.append(feature)
    
    return torch.stack(features) if features else torch.empty((0, feature_dim))

def cfg_to_pyg_data(code: str, label: Optional[int] = None) -> Data:
    """Convert code to CFG PyG Data"""
    extractor = CFGExtractor()
    nodes, edges = extractor.extract_cfg(code)
    
    x = create_cfg_features(nodes)
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)

def get_node_features_global(node_types, type2id, num_classes, target_dim=128):
    """Convert node types to feature vectors using global mappings"""
    indices = []
    for typ in node_types:
        if typ in type2id:
            indices.append(type2id[typ])
        else:
            indices.append(0)  # Unknown type
    
    if not indices:
        return torch.empty((0, target_dim))
    
    features = torch.nn.functional.one_hot(torch.tensor(indices), num_classes=num_classes).float()
    
    # Pad or truncate to target dimension
    if features.shape[1] < target_dim:
        pad_size = target_dim - features.shape[1]
        pad_tensor = torch.zeros(features.shape[0], pad_size)
        features = torch.cat([features, pad_tensor], dim=1)
    elif features.shape[1] > target_dim:
        features = features[:, :target_dim]
    
    return features

def check_sequence_length(ex):
    """Check if sequence fits within effective max length"""
    # Create the complete training sequence
    full_sequence = f"### Instruction:\n{ex['instruction']}\n\n### Input Code:\n{ex['input']}\n\n### Edited Code:\n{ex['output']}"
    
    # Tokenize to check length
    tokenized = tokenizer(full_sequence, add_special_tokens=True)
    seq_len = len(tokenized["input_ids"])
    
    return seq_len <= EFFECTIVE_MAX_LEN

def process_sample_correct_format(ex, ast_type2id, ast_num_classes, dfg_type2id, dfg_num_classes):
    """Process a single sample with correct training format"""
    
    # Create the complete training sequence
    full_sequence = f"### Instruction:\n{ex['instruction']}\n\n### Input Code:\n{ex['input']}\n\n### Edited Code:\n{ex['output']}"
    
    # Tokenize the complete sequence - use EFFECTIVE_MAX_LEN for text
    tokenized = tokenizer(full_sequence, truncation=True, padding="max_length", max_length=EFFECTIVE_MAX_LEN)
    
    # Create labels with proper masking
    prompt_part = f"### Instruction:\n{ex['instruction']}\n\n### Input Code:\n{ex['input']}\n\n### Edited Code:"
    prompt_tokens = tokenizer(prompt_part, add_special_tokens=False)["input_ids"]
    
    labels = tokenized["input_ids"].copy()
    # Mask the prompt part (model shouldn't predict these tokens)
    for i in range(min(len(prompt_tokens), len(labels))):
        labels[i] = -100
    
    # Build graphs from INPUT code (not output)
    input_code = ex["input"]
    
    # AST Graph
    ast_builder = ASTGraphBuilder()
    ast_node_types, ast_edges = ast_builder.build(input_code)
    
    if len(ast_node_types) > 0:
        x_ast = get_node_features_global(ast_node_types, ast_type2id, ast_num_classes)
        edge_index_ast = torch.tensor(ast_edges, dtype=torch.long).t().contiguous() if ast_edges else torch.empty((2, 0), dtype=torch.long)
        graph_ast = Data(x=x_ast, edge_index=edge_index_ast)
    else:
        graph_ast = Data(x=torch.empty((0, 128)), edge_index=torch.empty((2, 0), dtype=torch.long))
    
    # CFG Graph
    try:
        graph_cfg = cfg_to_pyg_data(input_code)
    except Exception:
        graph_cfg = Data(x=torch.empty((0, 128)), edge_index=torch.empty((2, 0), dtype=torch.long))
    
    # DFG Graph
    dfg_builder = DFGBuilder()
    dfg_node_types, dfg_edges = dfg_builder.build(input_code)
    
    if len(dfg_node_types) > 0:
        x_dfg = get_node_features_global(dfg_node_types, dfg_type2id, dfg_num_classes)
        edge_index_dfg = torch.tensor(dfg_edges, dtype=torch.long).t().contiguous() if dfg_edges else torch.empty((2, 0), dtype=torch.long)
        graph_dfg = Data(x=x_dfg, edge_index=edge_index_dfg)
    else:
        graph_dfg = Data(x=torch.empty((0, 128)), edge_index=torch.empty((2, 0), dtype=torch.long))
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
        "graph_ast": graph_ast,
        "graph_cfg": graph_cfg,
        "graph_dfg": graph_dfg,
    }

def analyze_sequence_lengths_with_filtering(ds, tokenizer, sample_size=5000):
    """Analyze sequence lengths and determine filtering statistics"""
    print(f"Analyzing sequence lengths with effective max length {EFFECTIVE_MAX_LEN}...")
    
    lengths = []
    skipped_count = 0
    kept_count = 0
    
    sample_indices = range(min(sample_size, len(ds)))
    
    for i in tqdm(sample_indices, desc="Analyzing sequences"):
        ex = ds[i]
        full_sequence = f"### Instruction:\n{ex['instruction']}\n\n### Input Code:\n{ex['input']}\n\n### Edited Code:\n{ex['output']}"
        
        tokens = tokenizer(full_sequence, add_special_tokens=True)
        seq_len = len(tokens["input_ids"])
        lengths.append(seq_len)
        
        if seq_len <= EFFECTIVE_MAX_LEN:
            kept_count += 1
        else:
            skipped_count += 1
    
    lengths = sorted(lengths)
    
    print(f"\nSequence Length Analysis:")
    print(f"Total analyzed: {len(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Mean length: {sum(lengths)/len(lengths):.1f}")
    print(f"Median length: {lengths[len(lengths)//2]}")
    
    print(f"\nFiltering Results:")
    print(f"Effective max length (text): {EFFECTIVE_MAX_LEN}")
    print(f"Reserved for graphs: {GRAPH_TOKEN_RESERVE}")
    print(f"Total model capacity: {MAX_LEN}")
    print(f"Sequences kept: {kept_count}/{len(lengths)} ({kept_count/len(lengths)*100:.1f}%)")
    print(f"Sequences skipped: {skipped_count}/{len(lengths)} ({skipped_count/len(lengths)*100:.1f}%)")
    
    return kept_count / len(lengths)

def main():
    print("Loading dataset...")
    ds = load_dataset("/home/xuhaoche/GACO/preprocessing/InstructCoder", split="train")
    
    print(f"\nConfiguration:")
    print(f"MAX_LEN: {MAX_LEN}")
    print(f"GRAPH_TOKEN_RESERVE: {GRAPH_TOKEN_RESERVE}")
    print(f"EFFECTIVE_MAX_LEN (for text): {EFFECTIVE_MAX_LEN}")
    
    # Analyze sequence lengths and filtering impact
    keep_ratio = analyze_sequence_lengths_with_filtering(ds, tokenizer)
    
    if keep_ratio < 0.7:  # If we're losing more than 30% of data
        print(f"\nWARNING: Only {keep_ratio*100:.1f}% of sequences fit within {EFFECTIVE_MAX_LEN} tokens.")
        print("Consider increasing MAX_LEN or reducing GRAPH_TOKEN_RESERVE if data loss is too high.")
        
        response = input("Continue with current settings? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\nCollecting global node type vocabularies...")
    
    # Collect AST node types (from all samples for vocabulary)
    global_ast_types = set()
    for ex in tqdm(ds, desc="Collecting AST types"):
        ast_builder = ASTGraphBuilder()
        node_types, _ = ast_builder.build(ex["input"])
        global_ast_types.update(node_types)
    
    global_ast_types = sorted(global_ast_types)
    ast_type2id = {typ: i for i, typ in enumerate(global_ast_types)}
    ast_num_classes = len(global_ast_types)
    print(f"Collected {ast_num_classes} unique AST node types")
    
    # Collect DFG node types (from all samples for vocabulary)
    global_dfg_types = set()
    for ex in tqdm(ds, desc="Collecting DFG types"):
        dfg_builder = DFGBuilder()
        node_types, _ = dfg_builder.build(ex["input"])
        global_dfg_types.update(node_types)
    
    global_dfg_types = sorted(global_dfg_types)
    dfg_type2id = {typ: i for i, typ in enumerate(global_dfg_types)}
    dfg_num_classes = len(global_dfg_types)
    print(f"Collected {dfg_num_classes} unique DFG node types")
    
    # Save feature mappings
    feature_mappings = {
        'ast_type2id': ast_type2id,
        'ast_num_classes': ast_num_classes,
        'dfg_type2id': dfg_type2id,
        'dfg_num_classes': dfg_num_classes,
        'target_dim': 128,
        'max_len': MAX_LEN,
        'effective_max_len': EFFECTIVE_MAX_LEN,
        'graph_token_reserve': GRAPH_TOKEN_RESERVE,
    }
    torch.save(feature_mappings, os.path.join(OUT_DIR, 'feature_mappings.pt'))
    
    # Process samples with length filtering
    print("Processing samples with length filtering...")
    processed = []
    skipped = 0
    
    for i, ex in enumerate(tqdm(ds, desc="Processing samples")):
        try:
            # Check sequence length first
            if not check_sequence_length(ex):
                skipped += 1
                continue
                
            sample = process_sample_correct_format(ex, ast_type2id, ast_num_classes, dfg_type2id, dfg_num_classes)
            processed.append(sample)
            
            # Debug info for first few samples
            if len(processed) <= 3:
                print(f"\nSample {len(processed)}:")
                print(f"  AST nodes: {sample['graph_ast'].x.shape[0]}")
                print(f"  CFG nodes: {sample['graph_cfg'].x.shape[0]}")
                print(f"  DFG nodes: {sample['graph_dfg'].x.shape[0]}")
                print(f"  Sequence length: {len(sample['input_ids'])}")
                print(f"  Non-masked labels: {sum(1 for x in sample['labels'] if x != -100)}")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Save processed data
    print(f"\nProcessing complete:")
    print(f"Total original samples: {len(ds)}")
    print(f"Skipped (too long): {skipped}")
    print(f"Successfully processed: {len(processed)}")
    print(f"Success rate: {len(processed)/(len(ds)-skipped)*100:.1f}%")
    
    torch.save(processed, os.path.join(OUT_DIR, 'processed_training_data.pt'))
    
    print(f"\nFiles saved in: {OUT_DIR}")
    print(f"- feature_mappings.pt: Node type vocabularies and config")
    print(f"- processed_training_data.pt: {len(processed)} filtered training samples")

if __name__ == "__main__":
    main()