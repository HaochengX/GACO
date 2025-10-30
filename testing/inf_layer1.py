import torch
import os
from models import LlamaWithGraphLayerSpecific  # Updated import
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from preprocessing import ASTGraphBuilder, CFGExtractor, DFGBuilder, cfg_to_pyg_data, process_sample_correct_format
from torch_geometric.data import Data

print("hello\n")

def load_trained_model_for_inference(checkpoint_path, model_path, target_layers=[0], device='cuda'):
    """
    Load the complete trained model for inference
    
    Args:
        checkpoint_path: Path to the checkpoint directory (e.g., "checkpoints_graph_lora/ckpt_step_1000")
        model_path: Path to the base LLaMA model
        target_layers: List of layer indices where graph integration is applied
        device: Device to load the model on
    """
    print("loading trained model for inference\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # CRITICAL FIX 1: Load base model first with target_layers parameter
    base_model = LlamaWithGraphLayerSpecific(
        llama_path=model_path,
        tokenizer=tokenizer,
        gnn_in_dim_ast=128,
        gnn_in_dim_cfg=128, 
        gnn_in_dim_dfg=128,
        target_layers=target_layers,  # NEW: specify target layers
        gnn_hid=256,
        gnn_out=256,
        graph_token_num=128,
        graph_hidden_dim=768
    )
    
    # CRITICAL FIX 2: Load LoRA weights correctly
    # First check if this is a LoRA checkpoint or full model checkpoint
    if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        # This is a LoRA checkpoint
        print("Loading LoRA checkpoint...")
        #r=12, alpha=16;
        # Apply LoRA config to base model
        lora_config = LoraConfig(
            r=12,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            use_rslora=True
        )
        base_model.llama = get_peft_model(base_model.llama, lora_config)
        
        # Load the LoRA weights
        base_model.llama = PeftModel.from_pretrained(
            base_model.llama.base_model, 
            checkpoint_path,
            is_trainable=False  # Set to inference mode
        )
        print("[Lora Adapter Loaded Successfully]")
    else:
        # This is a full model checkpoint
        print("Loading full model checkpoint...")
        checkpoint = torch.load(os.path.join(checkpoint_path, "model.pt"), map_location='cpu')
        base_model.load_state_dict(checkpoint, strict=False)
    
    # CRITICAL FIX 3: Load graph components properly - UPDATED FOR LAYER-SPECIFIC
    try:
        # First try to load layer-specific graph components
        graph_path = os.path.join(checkpoint_path, "graph_components_layerwise.pt")
        if os.path.exists(graph_path):
            base_model.load_graph_components(checkpoint_path)
            print("Layer-specific graph components loaded successfully")
        else:
            # Fallback to old format if available
            old_graph_path = os.path.join(checkpoint_path, "graph_components.pt")
            if os.path.exists(old_graph_path):
                print("Warning: Loading old format graph components")
                # You might need to implement compatibility loading here
                base_model.load_graph_components(checkpoint_path)
            else:
                print("Warning: No graph components found, using randomly initialized")
    except Exception as e:
        print(f"Warning: Could not load graph components: {e}")
    
    # Move to device and set to eval mode
    model = base_model.to(device)
    model = model.float()
    if hasattr(model, 'graph_tokens_cache'):
        model.graph_tokens_cache = None
    torch.cuda.empty_cache()
    model.eval()
    
    print(f"[Inference] Model loaded from {checkpoint_path}")
    print(f"[Inference] Graph integration active for layers: {target_layers}")
    return model, tokenizer

def get_ast_node_features_global(node_types, type2id, num_classes, target_dim):
    # print("getting ast node features\n")
    indices = []
    for typ in node_types:
        if typ in type2id:
            indices.append(type2id[typ])
        else:
            print(f"[Warning] Unknown node type: {typ}")
            indices.append(0)  # Map to unknown class
    
    features = torch.nn.functional.one_hot(torch.tensor(indices), num_classes=num_classes).float()
    
    # FIXED: Use the correct target_dim parameter
    if features.shape[1] < target_dim:
        pad_size = target_dim - features.shape[1]
        pad_tensor = torch.zeros(features.shape[0], pad_size)
        features = torch.cat([features, pad_tensor], dim=1)
    elif features.shape[1] > target_dim:
        features = features[:, :target_dim]
    
    return features

def get_dfg_node_features_global(node_types, dfg_type2id, dfg_num_classes, target_dim):
    # print("getting dfg node features\n")
    indices = [dfg_type2id.get(typ, 0) for typ in node_types]
    features = torch.nn.functional.one_hot(torch.tensor(indices), num_classes=dfg_num_classes).float()
    
    # FIXED: Use the correct target_dim parameter
    if features.shape[1] < target_dim:
        pad_size = target_dim - features.shape[1]
        pad_tensor = torch.zeros(features.shape[0], pad_size)
        features = torch.cat([features, pad_tensor], dim=1)
    elif features.shape[1] > target_dim:
        features = features[:, :target_dim]
    
    return features

def preprocess_inference_sample(code, instruction, ast_type2id, ast_num_classes, dfg_type2id, dfg_num_classes, target_dim, tokenizer, max_len=512):
    """
    Preprocess a single sample for inference using the same logic as training
    """
    print("start preprocessing inference sample\n")
    
    # CRITICAL FIX 4: Use exact same prompt format as training
    prompt = f"### Instruction:\n{instruction}\n\n### Input Code:\n{code}\n\n### Edited Code:"
    
    # Tokenize with consistent settings
    tok = tokenizer(
        prompt, 
        truncation=True, 
        padding="max_length", 
        max_length=max_len,
        return_tensors="pt"  # Return tensors directly
    )
    
    # AST Graph - with better error handling
    graph_ast = None
    try:
        ast_builder = ASTGraphBuilder()
        node_types, edge_list = ast_builder.build(code)
        
        if len(node_types) > 0:
            x_ast = get_ast_node_features_global(node_types, ast_type2id, ast_num_classes, target_dim)
            
            # Validate features
            if torch.isnan(x_ast).any() or torch.isinf(x_ast).any():
                print("Warning: AST features contain NaN/Inf, skipping AST graph")
                graph_ast = None
            else:
                edge_index_ast = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
                graph_ast = Data(x=x_ast, edge_index=edge_index_ast)
                print(f"AST graph created: {x_ast.shape[0]} nodes")
        else:
            print("AST graph is empty")
    except Exception as e:
        print(f"[AST] Error: {e}")
        graph_ast = None
    
    # CFG Graph
    graph_cfg = None
    try:
        graph_cfg = cfg_to_pyg_data(code, label=0)
        if graph_cfg is not None:
            # Validate CFG features
            if hasattr(graph_cfg, 'x') and graph_cfg.x is not None:
                if torch.isnan(graph_cfg.x).any() or torch.isinf(graph_cfg.x).any():
                    print("Warning: CFG features contain NaN/Inf, skipping CFG graph")
                    graph_cfg = None
                else:
                    print(f"CFG graph created: {graph_cfg.x.shape[0]} nodes")
            else:
                print("CFG graph has no features")
                graph_cfg = None
    except Exception as e:
        print(f"[CFG] Error: {e}")
        graph_cfg = None
    
    # DFG Graph
    graph_dfg = None
    try:
        dfg_builder = DFGBuilder()
        dfg_nodes, dfg_edges = dfg_builder.build(code)
        
        if len(dfg_nodes) > 0:
            x_dfg = get_dfg_node_features_global(dfg_nodes, dfg_type2id, dfg_num_classes, target_dim)
            
            # Validate features
            if torch.isnan(x_dfg).any() or torch.isinf(x_dfg).any():
                print("Warning: DFG features contain NaN/Inf, skipping DFG graph")
                graph_dfg = None
            else:
                edge_index_dfg = torch.tensor(dfg_edges, dtype=torch.long).t().contiguous() if dfg_edges else torch.empty((2, 0), dtype=torch.long)
                graph_dfg = Data(x=x_dfg, edge_index=edge_index_dfg)
                print(f"DFG graph created: {x_dfg.shape[0]} nodes")
        else:
            print("DFG graph is empty")
    except Exception as e:
        print(f"[DFG] Error: {e}")
        graph_dfg = None
    
    return {
        "input_ids": tok["input_ids"].squeeze(0),  # Remove batch dimension
        "attention_mask": tok["attention_mask"].squeeze(0),
        "graph_ast": graph_ast,
        "graph_cfg": graph_cfg,
        "graph_dfg": graph_dfg,
        "prompt": prompt
    }

def load_feature_mappings(output_dir):
    print("loading feature mappings\n")
    path = os.path.join(output_dir, 'feature_mappings.pt')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature mappings not found at {path}")
    
    feature_mappings = torch.load(path, map_location='cpu')
    
    # Extract into variables
    ast_type2id = feature_mappings['ast_type2id']
    ast_num_classes = feature_mappings['ast_num_classes']
    dfg_type2id = feature_mappings['dfg_type2id']
    dfg_num_classes = feature_mappings['dfg_num_classes']
    target_dim = feature_mappings['target_dim']
    
    print(f"Loaded mappings: AST classes={ast_num_classes}, DFG classes={dfg_num_classes}, target_dim={target_dim}")
    
    return ast_type2id, ast_num_classes, dfg_type2id, dfg_num_classes, target_dim

def generate_with_layerwise_graphs(model, tokenizer, input_ids, attention_mask, ast_batch, cfg_batch, dfg_batch, 
                                  max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=None):
    """
    OPTIMIZED: Separate prefill (with graphs) and generation (without graphs) phases
    - Prefill: Use full 512 token capacity with graph integration
    - Generation: Continue beyond 384 tokens without graph overhead
    """
    # print("DEBUG: generate_with_layerwise_graphs function called!")
    # print(f"DEBUG: input_ids type: {type(input_ids)}, shape: {input_ids.shape if hasattr(input_ids, 'shape') else 'no shape'}")
    
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # FIXED: Ensure input_ids is a tensor
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if isinstance(attention_mask, list):
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    
    # FIXED: Add missing variable definitions
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    # Store original input length for reference
    original_length = input_ids.shape[1]
    generated_ids = input_ids.clone()
    
    # print(f"DEBUG: Starting generation with sequence length: {generated_ids.shape[1]}")
    # print(f"DEBUG: Graph integration available: AST={ast_batch is not None}, CFG={cfg_batch is not None}, DFG={dfg_batch is not None}")
    
    # FIXED: Add missing unfinished_sequences tensor
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
    
    # PHASE 1: PREFILL with graph integration
    # print("DEBUG: PREFILL PHASE - Using graph integration")
    with torch.no_grad():
        try:
            # Initial forward pass WITH graphs (prefill phase)
            current_attention_mask = torch.ones_like(generated_ids, dtype=torch.long, device=device)
            
            prefill_outputs = model(
                input_ids=generated_ids,
                attention_mask=current_attention_mask,
                ast_batch=ast_batch,  # Use graphs in prefill
                cfg_batch=cfg_batch,
                dfg_batch=dfg_batch
            )
            
            # Extract logits for first generation step
            logits = prefill_outputs.logits if hasattr(prefill_outputs, 'logits') else prefill_outputs[0]
            next_token_logits = logits[:, -1, :]
            
            # print("DEBUG: Prefill phase successful with graph integration")
            
        except Exception as e:
            print(f"Error during prefill with graphs: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: prefill without graphs
            print("DEBUG: Retrying prefill without graphs...")
            try:
                prefill_outputs = model(
                    input_ids=generated_ids,
                    attention_mask=current_attention_mask,
                    ast_batch=None,
                    cfg_batch=None,
                    dfg_batch=None
                )
                logits = prefill_outputs.logits if hasattr(prefill_outputs, 'logits') else prefill_outputs[0]
                next_token_logits = logits[:, -1, :]
                print("DEBUG: Prefill successful without graphs")
                # Disable graphs for all subsequent steps
                ast_batch = cfg_batch = dfg_batch = None
            except Exception as e2:
                print(f"Failed prefill even without graphs: {e2}")
                return generated_ids
    
    # Generate first token from prefill
    if temperature != 1.0:
        next_token_logits = next_token_logits / temperature
    
    if do_sample:
        # Nucleus sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
        
        probs = torch.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_tokens = torch.argmax(next_token_logits, dim=-1)
    
    # print(f"Step 0 (PREFILL): Generated token {next_tokens[0].item()}")
    
    # Update sequences
    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
    generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
    
    # Check for EOS after first token
    unfinished_sequences = unfinished_sequences.mul(
        next_tokens.ne(tokenizer.eos_token_id).long()
    )
    
    if unfinished_sequences.max() == 0:
        # print("DEBUG: EOS token generated in prefill, stopping")
        return generated_ids
    
    # PHASE 2: AUTOREGRESSIVE GENERATION without graphs
    # print("DEBUG: GENERATION PHASE - No graph integration (unlimited length)")
    
    for step in range(1, max_new_tokens):  # Start from step 1 since step 0 was prefill
        # Use current generated sequence for attention
        current_attention_mask = torch.ones_like(generated_ids, dtype=torch.long, device=device)
        
        # Forward pass WITHOUT graphs (generation phase)
        with torch.no_grad():
            try:
                # NO GRAPHS during autoregressive generation
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=current_attention_mask,
                    ast_batch=None,  # No graphs in generation phase
                    cfg_batch=None,
                    dfg_batch=None
                )
                
                # Extract logits for next token
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                next_token_logits = logits[:, -1, :]
                
            except Exception as e:
                print(f"Error during generation step {step}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Generate next token
        if do_sample:
            # Nucleus sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)
        
        # print(f"Step {step} (GEN): Generated token {next_tokens[0].item()}")
        
        # Update sequences
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
        
        # print(f"Step {step}: Current sequence length: {generated_ids.shape[1]} (grew from {original_length})")
        
        # Check for EOS
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.ne(tokenizer.eos_token_id).long()
        )
        
        if unfinished_sequences.max() == 0:
            # print(f"Step {step}: EOS token generated, stopping")
            break
    
    final_length = generated_ids.shape[1]
    tokens_generated = final_length - original_length
    # print(f"DEBUG: Generation completed. Final length: {final_length} (+{tokens_generated} new tokens)")
    # print(f"DEBUG: Generation phases: PREFILL (with graphs) → AUTOREGRESSIVE (no graphs)")
    
    return generated_ids


def run_inference_with_layerwise_generation():
    """Complete inference example with layer-specific graph integration"""
    print("start running inference example with layer-specific generation\n")
    
    # Paths - VERIFY THESE PATHS ARE CORRECT
    checkpoint_path = "/home/xuhaoche/GACO/checkpoints_graph_lora/0910_layer1/final"
    model_path = "/home/xuhaoche/.llama/HF/Llama3.1-8B-Instruct"
    processed_data_dir = "processed_data/training_data"
    
    # Configuration for layer-specific integration
    target_layers = [0]  # You can modify this to match your training setup
    
    # Verify paths exist
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not os.path.exists(processed_data_dir):
        raise FileNotFoundError(f"Processed data dir does not exist: {processed_data_dir}")
    
    try:
        # 1. Load feature mappings
        ast_type2id, ast_num_classes, dfg_type2id, dfg_num_classes, target_dim = load_feature_mappings(processed_data_dir)
        
        # 2. Load trained model with layer-specific configuration
        model, tokenizer = load_trained_model_for_inference(
            checkpoint_path, 
            model_path, 
            target_layers=target_layers
        )
        
        # 3. Test with a simple example first
    #     test_code = """def fibonacci(n):
    # if n <= 1:
    #     return n
    # return fibonacci(n-1) + fibonacci(n-2)"""
        
    #     test_instruction = "Add error handling for negative numbers"
    #     # test_instruction = "Add error handling for numbers greater than 100"
        # test_instruction= "Create a new function to handle errors and call it whenever an exception occurs."
        # test_code = """import pandas as pd\n\ndef process_data(input_file, output_file):\n    try:\n        data = pd.read_csv(input_file)\n        # Data processing steps\n        processed_data = data.dropna().reset_index(drop=True)\n        processed_data.to_csv(output_file, index=False)\n    except Exception as e:\n        print(f\"Error occurred during data processing: {e}\")\n\ninput_file = \"data.csv\"\noutput_file = \"processed_data.csv\"\nprocess_data(input_file, output_file)"""
        # test_instruction = "Add rate limit handling to the get_repo_info method. If a 403 status code is returned with the message 'API rate limit exceeded', wait for the reset time and try again."
        # test_code = """import requests\nimport time\n\nclass GitHubAPI:\n    def __init__(self, access_token):\n        self.access_token = access_token\n\n    def get_repo_info(self, owner, repo_name):\n        headers = {\"Authorization\": f\"Bearer {self.access_token}\"}\n        url = f\"https://api.github.com/repos/{owner}/{repo_name}\"\n        response = requests.get(url, headers=headers)\n        if response.status_code == 200:\n            return response.json()\n        else:\n            return None\n\naccess_token = \"your_access_token_here\"\ngithub_api = GitHubAPI(access_token)\nrepo_info = github_api.get_repo_info(\"tensorflow\", \"tensorflow\")\nprint(repo_info)"""
       
        # 4. Preprocess
        test_sample = preprocess_inference_sample(
            test_code,
            test_instruction, 
            ast_type2id, ast_num_classes, dfg_type2id, dfg_num_classes, target_dim,
            tokenizer,
            max_len=128
        )
        
        print(f"Original prompt:\n{test_sample['prompt']}\n")
        print("="*50)
        
        # 5. Run inference with layer-specific graph integration
        with torch.no_grad():
            # Fix tensor construction warnings
            input_ids = test_sample["input_ids"].unsqueeze(0).cuda()
            attention_mask = test_sample["attention_mask"].unsqueeze(0).cuda()
            
            # Handle graph inputs with better validation
            ast_batch = test_sample["graph_ast"].to('cuda') if test_sample["graph_ast"] is not None else None
            cfg_batch = test_sample["graph_cfg"].to('cuda') if test_sample["graph_cfg"] is not None else None
            dfg_batch = test_sample["graph_dfg"].to('cuda') if test_sample["graph_dfg"] is not None else None
            
            # Debug: Print graph information
            print(f"Graph info:")
            print(f"  AST batch: {ast_batch is not None} - {ast_batch.x.shape if ast_batch is not None else 'None'}")
            print(f"  CFG batch: {cfg_batch is not None} - {cfg_batch.x.shape if cfg_batch is not None and hasattr(cfg_batch, 'x') else 'None'}")
            print(f"  DFG batch: {dfg_batch is not None} - {dfg_batch.x.shape if dfg_batch is not None else 'None'}")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Attention mask shape: {attention_mask.shape}")
            print(f"  Target layers for graph integration: {target_layers}")
            
            print("Starting generation with layer-specific graph integration...")
            
            # Generate with the layer-specific model
            generated_ids = generate_with_layerwise_graphs(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                ast_batch=ast_batch,
                cfg_batch=cfg_batch,
                dfg_batch=dfg_batch,
                max_new_tokens=256,  # Start with fewer tokens
                temperature=0.1,     # Lower temperature for more deterministic output
                top_p=0.95,
                do_sample=True
            )
            
            # Decode the response
            full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated_text = full_response[len(test_sample['prompt']):]
            
            print("GENERATED RESPONSE (with layer-specific graph integration):")
            print("="*50)
            print(generated_text)
            print("="*50)
            
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Layer-specific inference test\n")
    run_inference_with_layerwise_generation()