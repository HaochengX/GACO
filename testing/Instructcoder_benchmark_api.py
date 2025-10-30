# # import anthropic
# # import openai
# from google import genai
# from typing import List, Dict, Tuple
# import random
# import inspect
# import torch
# import os
# import json
# import gzip
# import time
# import ast
# import sys
# import subprocess
# import tempfile
# import re
# from tqdm import tqdm
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import LoraConfig, get_peft_model, PeftModel
# import numpy as np
# from collections import defaultdict
# from datetime import datetime
# from typing import Dict, List, Tuple, Any, Callable
# import difflib

# # Import your existing functions
# from models import LlamaWithGraphLayerSpecific
# from preprocessing import ASTGraphBuilder, CFGExtractor, DFGBuilder, cfg_to_pyg_data, process_sample_correct_format
# from torch_geometric.data import Data
# from testing import generate_with_layerwise_graphs, StandardBenchmarkEvaluator, CodeExtractor

# class LLMTestCaseGenerator:
#     """Use Claude/GPT to generate meaningful test cases"""
    
#     def __init__(self, provider: str = "gemini", api_key: str = None):
#         """
#         provider: "anthropic" or "openai" or gemini
#         api_key: Your API key (or set via environment variable)
#         """
#         self.provider = provider
        
#         if provider == "anthropic":
#             self.client = anthropic.Anthropic(
#                 api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
#             )
#             self.model = "claude-sonnet-4-20250514"  # Latest Claude
#         elif provider == "openai":
#             self.client = openai.OpenAI(
#                 api_key=api_key or os.environ.get("OPENAI_API_KEY")
#             )
#             self.model = "gpt-4o"  # GPT-4
#         elif provider == "gemini":
#             self.client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
#             self.model = "gemini-2.5-flash"
#         else:
#             raise ValueError("Provider must be 'anthropic', 'openai'" "genmini")
    
#     def generate_test_cases(self, instruction: str, ground_truth_code: str, 
#                            num_tests: int = 10) -> Dict:
#         """
#         Use LLM to analyze code and generate meaningful test cases
#         Returns: {
#             'test_cases': List[Dict],  # Each has 'input', 'expected_output', 'description'
#             'algorithm_type': str,      # BFS, DP, greedy, etc.
#             'edge_cases': List[str]     # What edge cases to test
#         }
#         """
        
#         prompt = f"""You are a expert software testing engineer. Analyze the following programming problem and solution, then generate comprehensive test cases.

# PROBLEM INSTRUCTION:
# {instruction}

# GROUND TRUTH SOLUTION:
# ```python
# {ground_truth_code}
# ```

# Your task:
# 1. Identify what type of algorithm this is (e.g., BFS, DFS, Dynamic Programming, Greedy, Two Pointers, etc.)
# 2. Identify the function name and its parameters
# 3. Generate {num_tests} diverse, meaningful test cases that cover:
#    - Basic/simple cases
#    - Edge cases (empty input, single element, maximum size, etc.)
#    - Corner cases (all same values, sorted/reverse sorted, etc.)
#    - Complex cases that test the core algorithm logic
   
# 4. For each test case, provide:
#    - Input values
#    - Expected output (run the code mentally or describe what it should return)
#    - Brief description of what this test case validates

# Return your answer in this EXACT JSON format:
# {{
#     "algorithm_type": "description of algorithm type",
#     "function_name": "name_of_function",
#     "parameters": ["param1_name", "param2_name"],
#     "test_cases": [
#         {{
#             "input": {{"param1_name": value1, "param2_name": value2}},
#             "expected_output": expected_value,
#             "description": "what this tests"
#         }}
#     ],
#     "edge_cases_covered": ["edge case 1", "edge case 2"]
# }}

# IMPORTANT: 
# - Make sure inputs are realistic and test different aspects of the algorithm
# - Expected outputs must be correct according to the ground truth code
# - Use diverse test cases, not just variations of the same thing
# - Return ONLY valid JSON, no other text
# """

#         if self.provider == "anthropic":
#             response = self.client.messages.create(
#                 model=self.model,
#                 max_tokens=4096,
#                 messages=[{"role": "user", "content": prompt}]
#             )
#             content = response.content[0].text
#         elif self.provider == "openai":  # openai
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.3
#             )
#             content = response.choices[0].message.content
#         elif self.provider == "gemini":
#             response = self.client.models.generate_content(
#                 model = self.model,
#                 contents = prompt
#             )
#             content = response.text
        
#         # Parse JSON from response
#         try:
#             # Try to find JSON in the response
#             import re
#             json_match = re.search(r'\{.*\}', content, re.DOTALL)
#             if json_match:
#                 result = json.loads(json_match.group())
#             else:
#                 result = json.loads(content)
            
#             return result
#         except json.JSONDecodeError as e:
#             print(f"Error parsing LLM response: {e}")
#             print(f"Response was: {content}")
#             return None
    
#     def create_executable_test_script(self, test_data: Dict, ground_truth_code: str) -> str:
#         """Convert LLM-generated test data into executable Python test script"""
        
#         if not test_data or 'test_cases' not in test_data:
#             return ""
        
#         function_name = test_data.get('function_name', 'solution')
#         test_cases = test_data['test_cases']
        
#         # Build test script
#         lines = [
#             "# Auto-generated test cases by LLM\n",
#             "# Algorithm type: " + test_data.get('algorithm_type', 'Unknown') + "\n",
#             "# Edge cases covered: " + ", ".join(test_data.get('edge_cases_covered', [])) + "\n\n",
#             "def check():\n",
#             "    test_results = []\n"
#         ]
        
#         for i, tc in enumerate(test_cases):
#             inputs = tc['input']
#             expected = tc['expected_output']
#             description = tc.get('description', '')
            
#             lines.append(f"\n    # Test {i+1}: {description}\n")
            
#             # Format function call
#             args = ', '.join(f"{k}={repr(v)}" for k, v in inputs.items())
            
#             lines.append(f"    try:\n")
#             lines.append(f"        result = {function_name}({args})\n")
#             lines.append(f"        expected = {repr(expected)}\n")
#             lines.append(f"        if result == expected:\n")
#             lines.append(f"            test_results.append(('Test {i+1}', 'PASS'))\n")
#             lines.append(f"        else:\n")
#             lines.append(f"            test_results.append(('Test {i+1}', f'FAIL: got {{result}}, expected {{expected}}'))\n")
#             lines.append(f"    except Exception as e:\n")
#             lines.append(f"        test_results.append(('Test {i+1}', f'ERROR: {{str(e)}}'))\n")
        
#         lines.append("\n    # Print results\n")
#         lines.append("    passed = sum(1 for _, status in test_results if status == 'PASS')\n")
#         lines.append("    total = len(test_results)\n")
#         lines.append("    print(f'Passed {passed}/{total} tests')\n")
#         lines.append("    for test_name, status in test_results:\n")
#         lines.append("        print(f'{test_name}: {status}')\n")
#         lines.append("    \n")
#         lines.append("    # Assert all passed\n")
#         lines.append("    assert passed == total, f'Only {passed}/{total} tests passed'\n\n")
        
#         lines.append("check()\n")
        
#         return ''.join(lines)


# class InstructCoderBenchmarkWithLLM:
#     """Enhanced InstructCoder benchmark with LLM-generated tests and graph contribution analysis"""
    
#     def __init__(self,
#                  baseline_model_path: str,
#                  graph_model_checkpoint_path: str,
#                  processed_data_dir: str,
#                  target_layers: List[int] = [0],
#                  device: str = 'cuda',
#                  num_samples: int = None,
#                  llm_provider: str = "gemini",
#                  llm_api_key: str = None,
#                  num_tests_per_problem: int = 5,
#                  save_code_files: bool = True,
#                  test_without_graphs: bool = True):  # NEW: Test graph model without graphs
        
#         self.baseline_model_path = baseline_model_path
#         self.graph_model_checkpoint_path = graph_model_checkpoint_path
#         self.processed_data_dir = processed_data_dir
#         self.target_layers = target_layers
#         self.device = device
#         self.num_samples = num_samples
#         self.num_tests_per_problem = num_tests_per_problem
#         self.save_code_files = save_code_files
#         self.test_without_graphs = test_without_graphs
        
#         self.evaluator = StandardBenchmarkEvaluator('instructcoder')
#         self.code_extractor = CodeExtractor()
#         self.test_generator = LLMTestCaseGenerator(provider=llm_provider, api_key=llm_api_key)
        
#         # Results storage - NOW WITH 3 MODEL VARIANTS
#         self.results = {
#             'baseline': [],                    # Baseline LLaMA
#             'graph_with_graphs': [],          # Graph model WITH graphs
#             'graph_without_graphs': [],       # Graph model WITHOUT graphs (measures model improvement)
#             'metadata': []
#         }
        
#         # Create output directories
#         if self.save_code_files:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             self.code_output_dir = f"generated_code_instructcoder_{timestamp}"
            
#             dirs_to_create = [
#                 "baseline",
#                 "graph_with_graphs",
#                 "graph_without_graphs",
#                 "tests",
#                 "ground_truth",
#                 "test_generation_logs"
#             ]
            
#             for dir_name in dirs_to_create:
#                 dir_path = os.path.join(self.code_output_dir, dir_name)
#                 os.makedirs(dir_path, exist_ok=True)
            
#             print(f"\n✓ All output directories created in: {self.code_output_dir}")
    
#     def load_instructcoder_dataset(self):
#         """Load InstructCoder validation set"""
#         from datasets import load_dataset
        
#         print("Loading InstructCoder dataset...")
#         dataset = load_dataset("/home/xuhaoche/GACO/preprocessing/InstructCoder", split="validation")
        
#         if self.num_samples:
#             dataset = dataset.select(range(min(self.num_samples, len(dataset))))
        
#         print(f"Loaded {len(dataset)} problems from InstructCoder validation set")
#         return dataset
    
#     def load_models(self):
#         """Load baseline and graph models"""
#         print("Loading models...")
        
#         # Load baseline
#         baseline_tokenizer = AutoTokenizer.from_pretrained(self.baseline_model_path)
#         if baseline_tokenizer.pad_token is None:
#             baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
        
#         baseline_model = AutoModelForCausalLM.from_pretrained(
#             self.baseline_model_path,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )
#         baseline_model.eval()
        
#         # Load graph model
#         graph_tokenizer = AutoTokenizer.from_pretrained(self.baseline_model_path)
#         if graph_tokenizer.pad_token is None:
#             graph_tokenizer.pad_token = graph_tokenizer.eos_token
        
#         graph_model = LlamaWithGraphLayerSpecific(
#             llama_path=self.baseline_model_path,
#             tokenizer=graph_tokenizer,
#             gnn_in_dim_ast=128,
#             gnn_in_dim_cfg=128,
#             gnn_in_dim_dfg=128,
#             target_layers=self.target_layers,
#             gnn_hid=256,
#             gnn_out=256,
#             graph_token_num=128,
#             graph_hidden_dim=768
#         )
        
#         # Load LoRA weights
#         if os.path.exists(os.path.join(self.graph_model_checkpoint_path, "adapter_config.json")):
#             lora_config = LoraConfig(
#                 r=12,
#                 lora_alpha=16,
#                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#                 lora_dropout=0.1,
#                 use_rslora=True
#             )
#             graph_model.llama = get_peft_model(graph_model.llama, lora_config)
#             graph_model.llama = PeftModel.from_pretrained(
#                 graph_model.llama.base_model,
#                 self.graph_model_checkpoint_path,
#                 is_trainable=False
#             )
        
#         # Load graph components
#         try:
#             graph_model.load_graph_components(self.graph_model_checkpoint_path)
#             print("Graph components loaded")
#         except Exception as e:
#             print(f"Warning: Could not load graph components: {e}")
        
#         graph_model = graph_model.to(self.device).float()
#         graph_model.eval()
        
#         # Load feature mappings
#         feature_path = os.path.join(self.processed_data_dir, 'feature_mappings.pt')
#         feature_mappings = torch.load(feature_path, map_location='cpu')
        
#         return (baseline_model, baseline_tokenizer,
#                 graph_model, graph_tokenizer,
#                 feature_mappings)
    
#     def create_prompt_from_instruction(self, instruction: str) -> str:
#         """Create prompt from instruction"""
#         prompt = f"""Write a Python function that solves the following problem:

# {instruction}

# Provide a complete, working implementation."""
#         return prompt
    
#     def generate_baseline(self, model, tokenizer, prompt: str,
#                          max_new_tokens: int = 512) -> str:
#         """Generate completion using baseline model"""
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.2,
#                 top_p=0.95,
#                 do_sample=True,
#                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
#             )
        
#         input_length = inputs['input_ids'].shape[1]
#         generated_tokens = outputs[0][input_length:]
#         response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
#         return response.strip()
    
#     def generate_graph(self, model, tokenizer, prompt: str,
#                       ground_truth_code: str,
#                       feature_mappings: Dict,
#                       use_graphs: bool = True,
#                       max_new_tokens: int = 512) -> str:
#         """
#         Generate completion using graph model
#         use_graphs: If False, test the model without graph input (to measure contribution)
#         """
        
#         # Build graphs from ground truth code (if using graphs)
#         if use_graphs:
#             ast_batch, cfg_batch, dfg_batch = self.build_graphs_from_code(
#                 ground_truth_code, feature_mappings
#             )
#         else:
#             ast_batch, cfg_batch, dfg_batch = None, None, None
        
#         # Tokenize
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
#         input_ids = inputs["input_ids"].to(self.device)
#         attention_mask = inputs["attention_mask"].to(self.device)
        
#         with torch.no_grad():
#             generated_ids = generate_with_layerwise_graphs(
#                 model=model,
#                 tokenizer=tokenizer,
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 ast_batch=ast_batch,
#                 cfg_batch=cfg_batch,
#                 dfg_batch=dfg_batch,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.2,
#                 top_p=0.95,
#                 do_sample=True
#             )
        
#         full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#         response = full_response[len(prompt):] if len(full_response) > len(prompt) else full_response
        
#         return response.strip()
    
#     def build_graphs_from_code(self, code: str, feature_mappings: Dict):
#         """Build graphs from ground truth code"""
#         try:
#             # Build AST
#             ast_builder = ASTGraphBuilder()
#             ast_nodes, ast_edges = ast_builder.build(code)
            
#             if len(ast_nodes) > 0:
#                 ast_type2id = feature_mappings['ast_type2id']
#                 ast_num_classes = feature_mappings['ast_num_classes']
#                 target_dim = feature_mappings['target_dim']
                
#                 indices = [ast_type2id.get(typ, 0) for typ in ast_nodes]
#                 x_ast = torch.nn.functional.one_hot(
#                     torch.tensor(indices),
#                     num_classes=ast_num_classes
#                 ).float()
                
#                 if x_ast.shape[1] < target_dim:
#                     pad = torch.zeros(x_ast.shape[0], target_dim - x_ast.shape[1])
#                     x_ast = torch.cat([x_ast, pad], dim=1)
#                 elif x_ast.shape[1] > target_dim:
#                     x_ast = x_ast[:, :target_dim]
                
#                 edge_index = torch.tensor(ast_edges, dtype=torch.long).t().contiguous() if ast_edges else torch.empty((2, 0), dtype=torch.long)
#                 ast_batch = Data(x=x_ast, edge_index=edge_index).to(self.device)
#             else:
#                 ast_batch = None
            
#             # Build CFG
#             try:
#                 cfg_batch = cfg_to_pyg_data(code).to(self.device)
#             except:
#                 cfg_batch = None
            
#             # Build DFG
#             dfg_builder = DFGBuilder()
#             dfg_nodes, dfg_edges = dfg_builder.build(code)
            
#             if len(dfg_nodes) > 0:
#                 dfg_type2id = feature_mappings['dfg_type2id']
#                 dfg_num_classes = feature_mappings['dfg_num_classes']
                
#                 indices = [dfg_type2id.get(typ, 0) for typ in dfg_nodes]
#                 x_dfg = torch.nn.functional.one_hot(
#                     torch.tensor(indices),
#                     num_classes=dfg_num_classes
#                 ).float()
                
#                 if x_dfg.shape[1] < target_dim:
#                     pad = torch.zeros(x_dfg.shape[0], target_dim - x_dfg.shape[1])
#                     x_dfg = torch.cat([x_dfg, pad], dim=1)
#                 elif x_dfg.shape[1] > target_dim:
#                     x_dfg = x_dfg[:, :target_dim]
                
#                 edge_index = torch.tensor(dfg_edges, dtype=torch.long).t().contiguous() if dfg_edges else torch.empty((2, 0), dtype=torch.long)
#                 dfg_batch = Data(x=x_dfg, edge_index=edge_index).to(self.device)
#             else:
#                 dfg_batch = None
            
#             return ast_batch, cfg_batch, dfg_batch
            
#         except Exception as e:
#             print(f"Warning: Could not build graphs from code: {e}")
#             return None, None, None
    
#     def run_benchmark(self):
#         """Run InstructCoder benchmark with LLM-generated tests"""
#         print("="*70)
#         print("INSTRUCTCODER BENCHMARK WITH LLM-GENERATED TESTS")
#         print("Testing 3 variants:")
#         print("  1. Baseline LLaMA")
#         print("  2. Graph Model WITH graphs (full system)")
#         print("  3. Graph Model WITHOUT graphs (measures model improvement only)")
#         print("="*70)
        
#         # Load dataset
#         dataset = self.load_instructcoder_dataset()
        
#         # Load models
#         (baseline_model, baseline_tokenizer,
#          graph_model, graph_tokenizer,
#          feature_mappings) = self.load_models()
        
#         start_time = time.time()
        
#         for i, sample in enumerate(tqdm(dataset, desc="Evaluating InstructCoder")):
#             try:
#                 instruction = sample['instruction']
#                 ground_truth = sample['output']
                
#                 print(f"\n{'='*70}")
#                 print(f"Problem {i}")
#                 print(f"Instruction: {instruction[:150]}...")
#                 print(f"{'='*70}")
                
#                 # Generate test cases using LLM
#                 print("🤖 Generating test cases using LLM...")
#                 test_data = self.test_generator.generate_test_cases(
#                     instruction=instruction,
#                     ground_truth_code=ground_truth,
#                     num_tests=self.num_tests_per_problem
#                 )
                
#                 if not test_data or 'test_cases' not in test_data:
#                     print(f"⚠️  Failed to generate test cases for problem {i}, skipping...")
#                     continue
                
#                 algorithm_type = test_data.get('algorithm_type', 'Unknown')
#                 num_tests = len(test_data['test_cases'])
#                 print(f"✓ Generated {num_tests} test cases")
#                 print(f"  Algorithm type: {algorithm_type}")
#                 print(f"  Edge cases: {', '.join(test_data.get('edge_cases_covered', []))}")
                
#                 # Save test generation log
#                 if self.save_code_files:
#                     log_path = os.path.join(
#                         self.code_output_dir, 
#                         "test_generation_logs",
#                         f"problem_{i:03d}_test_generation.json"
#                     )
#                     with open(log_path, 'w', encoding='utf-8') as f:
#                         json.dump(test_data, f, indent=2)
                
#                 # Create executable test script
#                 test_script = self.test_generator.create_executable_test_script(
#                     test_data, ground_truth
#                 )
                
#                 # Save ground truth with tests
#                 if self.save_code_files:
#                     gt_path = os.path.join(self.code_output_dir, "ground_truth", f"problem_{i:03d}_gt.py")
#                     with open(gt_path, 'w', encoding='utf-8') as f:
#                         f.write(f"# Ground Truth - Problem {i}\n")
#                         f.write(f"# Algorithm: {algorithm_type}\n")
#                         f.write("# " + "="*68 + "\n\n")
#                         f.write(ground_truth)
#                         f.write("\n\n# " + "="*68 + "\n")
#                         f.write("# LLM-generated tests\n")
#                         f.write("# " + "="*68 + "\n\n")
#                         f.write(test_script)
                
#                 # Create prompt
#                 prompt = self.create_prompt_from_instruction(instruction)
                
#                 # 1. Generate with BASELINE
#                 print("🔵 Generating with baseline model...")
#                 baseline_start = time.time()
#                 baseline_completion = self.generate_baseline(
#                     baseline_model, baseline_tokenizer, prompt
#                 )
#                 baseline_time = time.time() - baseline_start
                
#                 # 2. Generate with GRAPH MODEL (WITH graphs)
#                 print("🟢 Generating with graph model (WITH graphs)...")
#                 graph_with_start = time.time()
#                 graph_with_completion = self.generate_graph(
#                     graph_model, graph_tokenizer, prompt,
#                     ground_truth, feature_mappings,
#                     use_graphs=True
#                 )
#                 graph_with_time = time.time() - graph_with_start
                
#                 # 3. Generate with GRAPH MODEL (WITHOUT graphs)
#                 if self.test_without_graphs:
#                     print("🟡 Generating with graph model (WITHOUT graphs)...")
#                     graph_without_start = time.time()
#                     graph_without_completion = self.generate_graph(
#                         graph_model, graph_tokenizer, prompt,
#                         ground_truth, feature_mappings,
#                         use_graphs=False
#                     )
#                     graph_without_time = time.time() - graph_without_start
#                 else:
#                     graph_without_completion = ""
#                     graph_without_time = 0.0
                
#                 # Extract code
#                 baseline_code = self.code_extractor.simple_concatenate(baseline_completion, "")
#                 graph_with_code = self.code_extractor.simple_concatenate(graph_with_completion, "")
#                 graph_without_code = self.code_extractor.simple_concatenate(graph_without_completion, "") if self.test_without_graphs else ""
                
#                 # Check syntax
#                 baseline_syntax = self.evaluator.syntax_check(baseline_code)
#                 graph_with_syntax = self.evaluator.syntax_check(graph_with_code)
#                 graph_without_syntax = self.evaluator.syntax_check(graph_without_code) if self.test_without_graphs else False
                
#                 # Prepare test list
#                 test_list = [{'test': test_script}]
                
#                 # Evaluate all versions
#                 print("🧪 Running tests...")
#                 baseline_results = self.evaluator.evaluate_functional_correctness(
#                     baseline_code, test_list, timeout=10
#                 )
#                 graph_with_results = self.evaluator.evaluate_functional_correctness(
#                     graph_with_code, test_list, timeout=10
#                 )
                
#                 if self.test_without_graphs:
#                     graph_without_results = self.evaluator.evaluate_functional_correctness(
#                         graph_without_code, test_list, timeout=10
#                     )
#                 else:
#                     graph_without_results = {'passed': 0, 'failed': 0, 'error': 0, 'timeout': 0, 'pass_rate': 0.0}
                
#                 # Display results
#                 print(f"\n📊 Results:")
#                 print(f"  Baseline:             Syntax: {baseline_syntax}, Pass: {baseline_results['passed']}/{num_tests}, Time: {baseline_time:.2f}s")
#                 print(f"  Graph (WITH graphs):  Syntax: {graph_with_syntax}, Pass: {graph_with_results['passed']}/{num_tests}, Time: {graph_with_time:.2f}s")
#                 if self.test_without_graphs:
#                     print(f"  Graph (NO graphs):    Syntax: {graph_without_syntax}, Pass: {graph_without_results['passed']}/{num_tests}, Time: {graph_without_time:.2f}s")
                
#                 # Calculate graph contribution
#                 if self.test_without_graphs:
#                     model_improvement = graph_without_results['passed'] - baseline_results['passed']
#                     graph_contribution = graph_with_results['passed'] - graph_without_results['passed']
#                     print(f"\n📈 Analysis:")
#                     print(f"  Model improvement: {model_improvement:+d} tests (graph model architecture vs baseline)")
#                     print(f"  Graph contribution: {graph_contribution:+d} tests (adding graphs to graph model)")
#                     print(f"  Total improvement: {graph_with_results['passed'] - baseline_results['passed']:+d} tests")
                
#                 # Save generated code
#                 if self.save_code_files:
#                     for model_type, code, results, syntax, inf_time in [
#                         ("baseline", baseline_code, baseline_results, baseline_syntax, baseline_time),
#                         ("graph_with_graphs", graph_with_code, graph_with_results, graph_with_syntax, graph_with_time),
#                         ("graph_without_graphs", graph_without_code, graph_without_results, graph_without_syntax, graph_without_time)
#                     ]:
#                         if model_type == "graph_without_graphs" and not self.test_without_graphs:
#                             continue
                        
#                         self.save_generated_code_file(
#                             code=code,
#                             test=test_script,
#                             task_id=f"InstructCoder_{i}",
#                             model_type=model_type,
#                             problem_idx=i,
#                             test_results=results,
#                             syntax_valid=syntax,
#                             inference_time=inf_time,
#                             algorithm_type=algorithm_type
#                         )
                
#                 # Store results
#                 self.results['baseline'].append({
#                     'task_id': f"InstructCoder_{i}",
#                     'instruction': instruction,
#                     'algorithm_type': algorithm_type,
#                     'syntax_valid': baseline_syntax,
#                     'pass_rate': baseline_results['pass_rate'],
#                     'passed': baseline_results.get('passed', 0),
#                     'failed': baseline_results.get('failed', 0),
#                     'num_tests': num_tests,
#                     'inference_time': baseline_time,
#                     'code_length': len(baseline_code),
#                     'num_lines': len(baseline_code.splitlines()),
#                 })
                
#                 self.results['graph_with_graphs'].append({
#                     'task_id': f"InstructCoder_{i}",
#                     'instruction': instruction,
#                     'algorithm_type': algorithm_type,
#                     'syntax_valid': graph_with_syntax,
#                     'pass_rate': graph_with_results['pass_rate'],
#                     'passed': graph_with_results.get('passed', 0),
#                     'failed': graph_with_results.get('failed', 0),
#                     'num_tests': num_tests,
#                     'inference_time': graph_with_time,
#                     'code_length': len(graph_with_code),
#                     'num_lines': len(graph_with_code.splitlines()),
#                 })
                
#                 if self.test_without_graphs:
#                     self.results['graph_without_graphs'].append({
#                         'task_id': f"InstructCoder_{i}",
#                         'instruction': instruction,
#                         'algorithm_type': algorithm_type,
#                         'syntax_valid': graph_without_syntax,
#                         'pass_rate': graph_without_results['pass_rate'],
#                         'passed': graph_without_results.get('passed', 0),
#                         'failed': graph_without_results.get('failed', 0),
#                         'num_tests': num_tests,
#                         'inference_time': graph_without_time,
#                         'code_length': len(graph_without_code),
#                         'num_lines': len(graph_without_code.splitlines()),
#                     })
                
#                 self.results['metadata'].append({
#                     'task_id': f"InstructCoder_{i}",
#                     'instruction': instruction,
#                     'ground_truth': ground_truth,
#                     'algorithm_type': algorithm_type,
#                     'num_generated_tests': num_tests,
#                     'edge_cases_covered': test_data.get('edge_cases_covered', []),
#                     'model_improvement': model_improvement if self.test_without_graphs else None,
#                     'graph_contribution': graph_contribution if self.test_without_graphs else None
#                 })
                
#             except Exception as e:
#                 print(f"❌ Error on problem {i}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
        
#         total_time = time.time() - start_time
#         print(f"\n{'='*70}")
#         print(f"Benchmark completed in {total_time:.2f} seconds")
#         print(f"{'='*70}")
        
#         self.calculate_final_metrics()
#         self.save_results()
    
#     def save_generated_code_file(self, code: str, test: str, task_id: str,
#                                 model_type: str, problem_idx: int,
#                                 test_results: Dict = None,
#                                 syntax_valid: bool = True,
#                                 inference_time: float = 0.0,
#                                 algorithm_type: str = "Unknown"):
#         """Save generated code with enhanced metadata"""
#         if not self.save_code_files:
#             return
        
#         try:
#             safe_task_id = task_id.replace("/", "_").replace(" ", "_")
#             model_dir = os.path.join(self.code_output_dir, model_type)
            
#             if not os.path.exists(model_dir):
#                 os.makedirs(model_dir, exist_ok=True)
            
#             code_filename = f"problem_{problem_idx:03d}_{safe_task_id}.py"
#             code_path = os.path.join(model_dir, code_filename)
            
#             with open(code_path, 'w', encoding='utf-8') as f:
#                 f.write("# " + "="*70 + "\n")
#                 f.write(f"# Task ID: {task_id}\n")
#                 f.write(f"# Model: {model_type}\n")
#                 f.write(f"# Algorithm Type: {algorithm_type}\n")
#                 f.write(f"# Problem: {problem_idx}\n")
#                 f.write("# " + "="*70 + "\n")
#                 f.write("#\n")
#                 f.write("# EXECUTION RESULTS:\n")
#                 f.write(f"# Syntax Valid: {'✓ YES' if syntax_valid else '✗ NO'}\n")
                
#                 if test_results:
#                     passed = test_results.get('passed', 0)
#                     failed = test_results.get('failed', 0)
#                     errors = test_results.get('error', 0)
#                     timeouts = test_results.get('timeout', 0)
#                     pass_rate = test_results.get('pass_rate', 0.0)
                    
#                     f.write(f"# Test Results: {'✓ PASS' if passed > 0 else '✗ FAIL'}\n")
#                     f.write(f"#   - Passed: {passed}\n")
#                     f.write(f"#   - Failed: {failed}\n")
#                     f.write(f"#   - Errors: {errors}\n")
#                     f.write(f"#   - Timeouts: {timeouts}\n")
#                     f.write(f"#   - Pass Rate: {pass_rate:.1%}\n")
#                     f.write(f"# Inference Time: {inference_time:.3f}s\n")
#                     f.write("#\n")
                    
#                     if 'test_results' in test_results:
#                         f.write("# DETAILED TEST RESULTS:\n")
#                         for i, result in enumerate(test_results['test_results']):
#                             status = result['status']
#                             status_icon = "✓" if status == "passed" else "✗"
#                             f.write(f"# Test {i+1}: {status_icon} {status.upper()}\n")
                            
#                             if result.get('message'):
#                                 error_lines = result['message'].split('\n')
#                                 for line in error_lines[:5]:
#                                     if line.strip():
#                                         f.write(f"#   {line}\n")
#                         f.write("#\n")
#                 else:
#                     f.write("# Test Results: Not executed\n")
#                     f.write("#\n")
                
#                 f.write("# " + "="*70 + "\n\n")
#                 f.write(code)
#                 f.write("\n\n")
#                 f.write("# " + "="*70 + "\n")
#                 f.write("# LLM-GENERATED TEST CASES\n")
#                 f.write("# " + "="*70 + "\n")
#                 f.write("# Uncomment below to run tests:\n")
#                 for line in test.split('\n'):
#                     f.write(f"# {line}\n")
            
#             return code_path
            
#         except Exception as e:
#             print(f"ERROR saving file for {task_id}: {e}")
#             return None
    
#     def calculate_final_metrics(self):
#         """Calculate comprehensive metrics including graph contribution analysis"""
#         print("\n" + "="*70)
#         print("INSTRUCTCODER BENCHMARK RESULTS - GRAPH CONTRIBUTION ANALYSIS")
#         print("="*70)
        
#         if not self.results.get('baseline'):
#             print("No results to display")
#             return
        
#         total = len(self.results['baseline'])
        
#         # Syntax validity
#         baseline_syntax = np.mean([r['syntax_valid'] for r in self.results['baseline']])
#         graph_with_syntax = np.mean([r['syntax_valid'] for r in self.results['graph_with_graphs']])
        
#         print(f"\nSYNTAX VALIDITY:")
#         print(f"  Baseline:            {baseline_syntax:.3f}")
#         print(f"  Graph (with graphs): {graph_with_syntax:.3f}")
        
#         if self.test_without_graphs and self.results.get('graph_without_graphs'):
#             graph_without_syntax = np.mean([r['syntax_valid'] for r in self.results['graph_without_graphs']])
#             print(f"  Graph (no graphs):   {graph_without_syntax:.3f}")
        
#         # Pass rates
#         baseline_pass = np.mean([r['pass_rate'] for r in self.results['baseline']])
#         graph_with_pass = np.mean([r['pass_rate'] for r in self.results['graph_with_graphs']])
        
#         print(f"\nPASS@1:")
#         print(f"  Baseline:            {baseline_pass:.3f}")
#         print(f"  Graph (with graphs): {graph_with_pass:.3f}")
        
#         if self.test_without_graphs and self.results.get('graph_without_graphs'):
#             graph_without_pass = np.mean([r['pass_rate'] for r in self.results['graph_without_graphs']])
#             print(f"  Graph (no graphs):   {graph_without_pass:.3f}")
            
#             # DECOMPOSITION OF IMPROVEMENT
#             print(f"\n🔍 IMPROVEMENT DECOMPOSITION:")
#             total_improvement = graph_with_pass - baseline_pass
#             model_improvement = graph_without_pass - baseline_pass
#             graph_contribution = graph_with_pass - graph_without_pass
            
#             print(f"  Total improvement:     {total_improvement:+.3f} ({(total_improvement/max(baseline_pass,0.001)*100):+.1f}%)")
#             print(f"    ├─ Model improvement: {model_improvement:+.3f} ({(model_improvement/max(total_improvement,0.001)*100):.1f}% of total)")
#             print(f"    └─ Graph contribution: {graph_contribution:+.3f} ({(graph_contribution/max(total_improvement,0.001)*100):.1f}% of total)")
            
#             if graph_contribution > 0:
#                 print(f"\n  ✅ Graphs provide {graph_contribution:.3f} improvement!")
#                 print(f"     This is {(graph_contribution/total_improvement*100):.1f}% of the total improvement")
#             elif graph_contribution < 0:
#                 print(f"\n  ⚠️  Graphs actually HURT performance by {abs(graph_contribution):.3f}")
#                 print(f"     The model improvement alone is {model_improvement:.3f}")
#             else:
#                 print(f"\n  ⚪ Graphs have no measurable effect")
#         else:
#             total_improvement = graph_with_pass - baseline_pass
#             print(f"\nIMPROVEMENT:")
#             print(f"  Graph vs Baseline: {total_improvement:+.3f} ({(total_improvement/max(baseline_pass,0.001)*100):+.1f}%)")
        
#         # Perfect solutions
#         baseline_perfect = sum(1 for r in self.results['baseline'] if r['pass_rate'] == 1.0)
#         graph_with_perfect = sum(1 for r in self.results['graph_with_graphs'] if r['pass_rate'] == 1.0)
        
#         print(f"\nPERFECT SOLUTIONS:")
#         print(f"  Baseline:            {baseline_perfect}/{total} ({baseline_perfect/total*100:.1f}%)")
#         print(f"  Graph (with graphs): {graph_with_perfect}/{total} ({graph_with_perfect/total*100:.1f}%)")
        
#         if self.test_without_graphs and self.results.get('graph_without_graphs'):
#             graph_without_perfect = sum(1 for r in self.results['graph_without_graphs'] if r['pass_rate'] == 1.0)
#             print(f"  Graph (no graphs):   {graph_without_perfect}/{total} ({graph_without_perfect/total*100:.1f}%)")
        
#         # Algorithm type breakdown
#         print(f"\nPERFORMANCE BY ALGORITHM TYPE:")
#         algo_types = {}
#         for i, meta in enumerate(self.results['metadata']):
#             algo_type = meta.get('algorithm_type', 'Unknown')
#             if algo_type not in algo_types:
#                 algo_types[algo_type] = {
#                     'baseline': [],
#                     'graph_with': [],
#                     'graph_without': []
#                 }
            
#             if i < len(self.results['baseline']):
#                 algo_types[algo_type]['baseline'].append(self.results['baseline'][i]['pass_rate'])
#             if i < len(self.results['graph_with_graphs']):
#                 algo_types[algo_type]['graph_with'].append(self.results['graph_with_graphs'][i]['pass_rate'])
#             if self.test_without_graphs and i < len(self.results.get('graph_without_graphs', [])):
#                 algo_types[algo_type]['graph_without'].append(self.results['graph_without_graphs'][i]['pass_rate'])
        
#         for algo_type, scores in sorted(algo_types.items()):
#             if scores['baseline']:
#                 baseline_avg = np.mean(scores['baseline'])
#                 graph_with_avg = np.mean(scores['graph_with'])
                
#                 print(f"\n  {algo_type} (n={len(scores['baseline'])}):")
#                 print(f"    Baseline:            {baseline_avg:.3f}")
#                 print(f"    Graph (with graphs): {graph_with_avg:.3f} ({(graph_with_avg-baseline_avg):+.3f})")
                
#                 if self.test_without_graphs and scores['graph_without']:
#                     graph_without_avg = np.mean(scores['graph_without'])
#                     print(f"    Graph (no graphs):   {graph_without_avg:.3f}")
#                     print(f"    Graph contribution:  {(graph_with_avg-graph_without_avg):+.3f}")
        
#         # Test statistics
#         avg_tests = np.mean([r['num_tests'] for r in self.results['baseline']])
#         print(f"\nAVG LLM-GENERATED TESTS PER PROBLEM: {avg_tests:.1f}")
        
#         # Inference time
#         baseline_time = np.mean([r['inference_time'] for r in self.results['baseline']])
#         graph_with_time = np.mean([r['inference_time'] for r in self.results['graph_with_graphs']])
        
#         print(f"\nAVERAGE INFERENCE TIME:")
#         print(f"  Baseline:            {baseline_time:.3f}s")
#         print(f"  Graph (with graphs): {graph_with_time:.3f}s ({(graph_with_time-baseline_time):+.3f}s, {((graph_with_time-baseline_time)/baseline_time*100):+.1f}%)")
        
#         if self.test_without_graphs and self.results.get('graph_without_graphs'):
#             graph_without_time = np.mean([r['inference_time'] for r in self.results['graph_without_graphs']])
#             print(f"  Graph (no graphs):   {graph_without_time:.3f}s")
    
#     def save_results(self):
#         """Save comprehensive results"""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_dir = f"instructcoder_results_{timestamp}"
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Save raw results
#         with open(os.path.join(output_dir, "results.json"), "w") as f:
#             json.dump(self.results, f, indent=2, default=str)
        
#         # Create summary with graph contribution analysis
#         with open(os.path.join(output_dir, "summary.txt"), "w") as f:
#             f.write("INSTRUCTCODER BENCHMARK WITH GRAPH CONTRIBUTION ANALYSIS\n")
#             f.write("="*60 + "\n\n")
            
#             total = len(self.results.get('baseline', []))
#             f.write(f"Total problems: {total}\n\n")
            
#             if total > 0:
#                 baseline_pass = np.mean([r['pass_rate'] for r in self.results['baseline']])
#                 graph_with_pass = np.mean([r['pass_rate'] for r in self.results['graph_with_graphs']])
                
#                 f.write("PASS@1 SCORES:\n")
#                 f.write(f"  Baseline:            {baseline_pass:.3f}\n")
#                 f.write(f"  Graph (with graphs): {graph_with_pass:.3f}\n")
                
#                 if self.test_without_graphs and self.results.get('graph_without_graphs'):
#                     graph_without_pass = np.mean([r['pass_rate'] for r in self.results['graph_without_graphs']])
#                     f.write(f"  Graph (no graphs):   {graph_without_pass:.3f}\n\n")
                    
#                     f.write("IMPROVEMENT DECOMPOSITION:\n")
#                     total_improvement = graph_with_pass - baseline_pass
#                     model_improvement = graph_without_pass - baseline_pass
#                     graph_contribution = graph_with_pass - graph_without_pass
                    
#                     f.write(f"  Total improvement:     {total_improvement:+.3f}\n")
#                     f.write(f"  Model improvement:     {model_improvement:+.3f} ({(model_improvement/max(total_improvement,0.001)*100):.1f}% of total)\n")
#                     f.write(f"  Graph contribution:    {graph_contribution:+.3f} ({(graph_contribution/max(total_improvement,0.001)*100):.1f}% of total)\n\n")
                    
#                     if graph_contribution > 0:
#                         f.write(f"✅ Graphs contribute {(graph_contribution/total_improvement*100):.1f}% of the improvement\n")
#                     elif graph_contribution < 0:
#                         f.write(f"⚠️  Graphs hurt performance\n")
#                 else:
#                     f.write(f"\nTotal improvement: {(graph_with_pass - baseline_pass):+.3f}\n")
                
#                 avg_tests = np.mean([r['num_tests'] for r in self.results['baseline']])
#                 f.write(f"\nAverage LLM-generated tests per problem: {avg_tests:.1f}\n")
        
#         print(f"\n✓ Results saved to {output_dir}/")
#         if self.save_code_files:
#             print(f"✓ Generated code saved to {self.code_output_dir}/")


# def run_instructcoder_benchmark_with_llm():
#     """Main function"""
    
#     # Set your API key
#     # Option 1: Set environment variable ANTHROPIC_API_KEY or OPENAI_API_KEY
#     # Option 2: Pass it directly
    
#     benchmark = InstructCoderBenchmarkWithLLM(
#         baseline_model_path="/home/xuhaoche/.llama/HF/Llama3.1-8B-Instruct",
#         graph_model_checkpoint_path="/home/xuhaoche/GACO/checkpoints_graph_lora/0910_layer1/final",
#         processed_data_dir="processed_data/training_data",
#         target_layers=[0],
#         device='cuda',
#         num_samples=100,  # Test on 50 samples
#         llm_provider="gemini",  # or "openai" "anthropic" "gemini"
#         llm_api_key="AIzaSyDtr2GqqhJXSkp_SJ5StJN6JGp3tA8QEHo",  # or set environment variable
#         num_tests_per_problem=3,
#         save_code_files=True,
#         test_without_graphs=True  # IMPORTANT: Tests graph model both with and without graphs
#     )
    
#     benchmark.run_benchmark()


# if __name__ == "__main__":
#     run_instructcoder_benchmark_with_llm()
# # ```

# # **Key improvements:**

# # 1. **LLM-Generated Test Cases**: Uses Claude/GPT-4 to analyze the problem and generate meaningful, diverse test cases that actually test the algorithm logic

# # 2. **Graph Contribution Analysis**: Tests 3 variants:
# #    - **Baseline**: Regular LLaMA
# #    - **Graph (with graphs)**: Your full system
# #    - **Graph (without graphs)**: Graph model but NO graph input
   
# #    This lets you decompose the improvement into:
# #    - Model improvement (architecture/training changes)
# #    - Graph contribution (actual benefit from graphs)

# # 3. **Algorithm-Type Breakdown**: Shows performance by algorithm type (BFS, DP, etc.)

# # 4. **Better Test Quality**: LLM understands edge cases, complex cases, etc.

# # The metrics will show you exactly how much the graphs contribute! For example:
# # ```
# # Total improvement: +0.15
# #   ├─ Model improvement: +0.08 (53% of total)
# #   └─ Graph contribution: +0.07 (47% of total)

# import anthropic
# import openai
from google import genai
from typing import List, Dict, Tuple
import random
import inspect
import torch
import os
import json
import gzip
import time
import ast
import sys
import subprocess
import tempfile
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Callable
import difflib

# Import your existing functions
from models import LlamaWithGraphLayerSpecific
from preprocessing import ASTGraphBuilder, CFGExtractor, DFGBuilder, cfg_to_pyg_data, process_sample_correct_format
from torch_geometric.data import Data
from testing import generate_with_layerwise_graphs, StandardBenchmarkEvaluator, CodeExtractor
class GeminiTestGenerator:
    """Use Gemini to generate meaningful test cases for code problems"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini test generator
        
        Args:
            api_key: Your Google AI API key
            model_name: Gemini model to use (gemini-1.5-flash )
        """
        self.client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.model = "gemini-2.5-flash"
    
    def generate_test_cases(self, 
                          instruction: str, 
                          ground_truth_code: str,
                          num_tests: int = 3) -> Tuple[str, List[Dict]]:
        """
        Generate meaningful test cases using Gemini
        
        Returns:
            (test_script, parsed_test_cases)
        """
        
        prompt = f"""You are an expert Python testing engineer. Given a programming problem and its solution, generate comprehensive test cases.

**Problem Description:**
{instruction}

**Ground Truth Solution:**
```python
{ground_truth_code}
```

**Your Task:**
Generate EXACTLY {num_tests} separate test cases that thoroughly test this function. Each test case should be independent. Include:
1. Edge cases (empty inputs, None, zeros, negatives)
2. Normal cases (typical inputs)
3. Boundary cases (min/max values, large inputs)
4. Corner cases specific to the algorithm (e.g., for BFS: disconnected graphs, cycles; for DP: base cases)

**IMPORTANT**: Create exactly {num_tests} distinct test cases. Each test should have ONE assertion.

**Output Format:**
Provide ONLY executable Python code with this exact structure:
```python
def check():
    # Test 1: [Description of what this tests]
    result = function_name(input1)
    expected = expected_value
    assert result == expected, f"Test 1 failed: {{result}} != {{expected}}"
    
    # Test 2: [Description]
    result = function_name(input2)
    expected = expected_value
    assert result == expected, f"Test 2 failed: {{result}} != {{expected}}"
    
    # ... continue for all {num_tests} tests ...

check()
print("All tests passed!")
```

Generate the complete test code now with EXACTLY {num_tests} test cases:"""

        try:
            response = self.client.models.generate_content(
                model = self.model,
                contents = prompt
            )
            test_code = response.text
            
            # Extract code from markdown if present
            if '```python' in test_code:
                test_code = re.findall(r'```python\s*\n(.*?)\n```', test_code, re.DOTALL)[0]
            elif '```' in test_code:
                test_code = re.findall(r'```\s*\n(.*?)\n```', test_code, re.DOTALL)[0]
            
            # Parse test cases for metadata
            test_cases = self._parse_test_cases(test_code)
            
            return test_code.strip(), test_cases
            
        except Exception as e:
            print(f"Error generating test cases with Gemini: {e}")
            return "", []
    
    def _parse_test_cases(self, test_code: str) -> List[Dict]:
        """Parse generated test code to extract test case metadata"""
        test_cases = []
        
        # Find all comments that indicate test cases (# Test 1:, # Test 2:, etc.)
        test_markers = re.findall(r'#\s*Test\s+(\d+):\s*(.*?)(?=\n)', test_code, re.IGNORECASE)
        
        # If we found test markers, use those
        if test_markers:
            for test_num, description in test_markers:
                test_cases.append({
                    'test_id': int(test_num),
                    'description': description.strip()
                })
        else:
            # Fallback: count assertions
            assertions = re.findall(r'assert\s+.*?(?=\n|$)', test_code)
            for i, assertion in enumerate(assertions):
                test_cases.append({
                    'test_id': i + 1,
                    'assertion': assertion.strip()
                })
        
        return test_cases
    
    def validate_test_code(self, test_code: str, ground_truth_code: str) -> Tuple[bool, str]:
        """
        Validate that generated test code works with ground truth
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Combine ground truth and test code
            full_code = ground_truth_code + "\n\n" + test_code
            
            # Try to execute
            namespace = {}
            exec(full_code, namespace)
            
            return True, ""
            
        except Exception as e:
            return False, str(e)



# class GeminiTestGenerator:
#     """Use Gemini to generate meaningful test cases for code problems"""
    
#     def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
#         """
#         Initialize Gemini test generator
        
#         Args:
#             api_key: Your Google AI API key
#             model_name: Gemini model to use (gemini-1.5-flash or gemini-1.5-pro)
#         """
#         # genai.configure(api_key=api_key)
#         # self.model = model_name
#         # self.generation_config = {
#         #     "temperature": 0.7,
#         #     "top_p": 0.95,
#         #     "top_k": 40,
#         #     # "max_output_tokens": 4096,
#         # }
#         self.client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
#         self.model = "gemini-2.5-flash"
    
#     def generate_test_cases(self, 
#                           instruction: str, 
#                           ground_truth_code: str,
#                           num_tests: int = 5) -> Tuple[str, List[Dict]]:
#         """
#         Generate meaningful test cases using Gemini
        
#         Returns:
#             (test_script, parsed_test_cases)
#         """
        
#         prompt = f"""You are an expert Python testing engineer. Given a programming problem and its solution, generate comprehensive test cases.

# **Problem Description:**
# {instruction}

# **Ground Truth Solution:**
# ```python
# {ground_truth_code}
# ```

# **Your Task:**
# Generate {num_tests} diverse test cases that thoroughly test this function. Include:
# 1. Edge cases (empty inputs, None, zeros, negatives)
# 2. Normal cases (typical inputs)
# 3. Boundary cases (min/max values, large inputs)
# 4. Corner cases specific to the algorithm (e.g., for BFS: disconnected graphs, cycles; for DP: base cases)

# **Output Format:**
# Provide ONLY executable Python code that:
# 1. Imports the necessary function from the ground truth
# 2. Defines a `check()` function that runs all test cases
# 3. Each test case should use `assert` statements
# 4. Add descriptive comments for each test

# Example structure:
# ```python
# def check():
#     # Test 1: Edge case - empty input
#     result = function_name([])
#     expected = expected_value
#     assert result == expected, f"Test 1 failed: {{result}} != {{expected}}"
    
#     # Test 2: Normal case
#     result = function_name([1, 2, 3])
#     expected = expected_value
#     assert result == expected, f"Test 2 failed: {{result}} != {{expected}}"
    
#     # ... more tests ...

# check()
# print("All tests passed!")
# ```

# Generate the complete test code now:"""

#         try:
#             response = self.client.models.generate_content(
#                 model = self.model,
#                 contents = prompt
#             )
#             test_code = response.text
            
#             # Extract code from markdown if present
#             if '```python' in test_code:
#                 test_code = re.findall(r'```python\s*\n(.*?)\n```', test_code, re.DOTALL)[0]
#             elif '```' in test_code:
#                 test_code = re.findall(r'```\s*\n(.*?)\n```', test_code, re.DOTALL)[0]
            
#             # Parse test cases for metadata
#             test_cases = self._parse_test_cases(test_code)
            
#             return test_code.strip(), test_cases
            
#         except Exception as e:
#             print(f"Error generating test cases with Gemini: {e}")
#             return "", []
    
#     def _parse_test_cases(self, test_code: str) -> List[Dict]:
#         """Parse generated test code to extract test case metadata"""
#         test_cases = []
        
#         # Count assertions
#         assertions = re.findall(r'assert\s+.*?(?=\n|$)', test_code)
        
#         for i, assertion in enumerate(assertions):
#             test_cases.append({
#                 'test_id': i + 1,
#                 'assertion': assertion.strip()
#             })
        
#         return test_cases
    
#     def validate_test_code(self, test_code: str, ground_truth_code: str) -> Tuple[bool, str]:
#         """
#         Validate that generated test code works with ground truth
        
#         Returns:
#             (is_valid, error_message)
#         """
#         try:
#             # Combine ground truth and test code
#             full_code = ground_truth_code + "\n\n" + test_code
            
#             # Try to execute
#             namespace = {}
#             exec(full_code, namespace)
            
#             return True, ""
            
#         except Exception as e:
#             return False, str(e)


class InstructCoderBenchmarkWithGemini:
    """InstructCoder benchmark with Gemini-generated tests and graph analysis"""
    
    def __init__(self,
                 baseline_model_path: str,
                 graph_model_checkpoint_path: str,
                 processed_data_dir: str,
                 gemini_api_key: str,
                 target_layers: List[int] = [0],
                 device: str = 'cuda',
                 num_samples: int = None,
                 num_tests_per_problem: int = 10,
                 save_code_files: bool = True):
        
        self.baseline_model_path = baseline_model_path
        self.graph_model_checkpoint_path = graph_model_checkpoint_path
        self.processed_data_dir = processed_data_dir
        self.target_layers = target_layers
        self.device = device
        self.num_samples = num_samples
        self.num_tests_per_problem = num_tests_per_problem
        self.save_code_files = save_code_files
        
        self.evaluator = StandardBenchmarkEvaluator('instructcoder')
        self.code_extractor = CodeExtractor()
        self.test_generator = GeminiTestGenerator(gemini_api_key)
        
        # Results storage - 3 models x 2 extraction methods = 6 versions
        self.results = {
            'baseline_simple': [],
            'baseline_clean': [],
            'graph_with_graphs_simple': [],
            'graph_with_graphs_clean': [],
            'graph_no_graphs_simple': [],
            'graph_no_graphs_clean': [],
            'test_cases': [],  # Store all generated test cases
            'metadata': []
        }
        
        # Create output directories
        if self.save_code_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.code_output_dir = f"instructcoder_gemini_{timestamp}"
            
            dirs_to_create = [
                "baseline_simple",
                "baseline_clean",
                "graph_with_graphs_simple",
                "graph_with_graphs_clean",
                "graph_no_graphs_simple",
                "graph_no_graphs_clean",
                "tests",
                "ground_truth",
                "generated_test_cases"
            ]
            
            for dir_name in dirs_to_create:
                dir_path = os.path.join(self.code_output_dir, dir_name)
                os.makedirs(dir_path, exist_ok=True)
            
            print(f"\n✓ All output directories created in: {self.code_output_dir}")
    
    def load_instructcoder_dataset(self):
        """Load InstructCoder validation set"""
        from datasets import load_dataset
        
        print("Loading InstructCoder dataset...")
        dataset = load_dataset("/home/xuhaoche/GACO/preprocessing/InstructCoder", split="validation")
        
        if self.num_samples:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))
        
        print(f"Loaded {len(dataset)} problems from InstructCoder validation set")
        return dataset
    
    def create_prompt_from_instruction(self, instruction: str) -> str:
        """Create a prompt from instruction"""
        prompt = f"""Complete the following Python function:\n

{instruction}
"""
        
        return prompt
    
    def load_models(self):
        """Load baseline and graph models"""
        print("Loading models...")
        
        # Load baseline
        baseline_tokenizer = AutoTokenizer.from_pretrained(self.baseline_model_path)
        if baseline_tokenizer.pad_token is None:
            baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
        
        baseline_model = AutoModelForCausalLM.from_pretrained(
            self.baseline_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        baseline_model.eval()
        
        # Load graph model
        graph_tokenizer = AutoTokenizer.from_pretrained(self.baseline_model_path)
        if graph_tokenizer.pad_token is None:
            graph_tokenizer.pad_token = graph_tokenizer.eos_token
        
        graph_model = LlamaWithGraphLayerSpecific(
            llama_path=self.baseline_model_path,
            tokenizer=graph_tokenizer,
            gnn_in_dim_ast=128,
            gnn_in_dim_cfg=128,
            gnn_in_dim_dfg=128,
            target_layers=self.target_layers,
            gnn_hid=256,
            gnn_out=256,
            graph_token_num=128,
            graph_hidden_dim=768
        )
        
        # Load LoRA weights
        if os.path.exists(os.path.join(self.graph_model_checkpoint_path, "adapter_config.json")):
            lora_config = LoraConfig(
                r=12,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.1,
                use_rslora=True
            )
            graph_model.llama = get_peft_model(graph_model.llama, lora_config)
            graph_model.llama = PeftModel.from_pretrained(
                graph_model.llama.base_model,
                self.graph_model_checkpoint_path,
                is_trainable=False
            )
        
        # Load graph components
        try:
            graph_model.load_graph_components(self.graph_model_checkpoint_path)
            print("✓ Graph components loaded")
        except Exception as e:
            print(f"Warning: Could not load graph components: {e}")
        
        graph_model = graph_model.to(self.device).float()
        graph_model.eval()
        
        # Load feature mappings
        feature_path = os.path.join(self.processed_data_dir, 'feature_mappings.pt')
        feature_mappings = torch.load(feature_path, map_location='cpu')
        
        return (baseline_model, baseline_tokenizer,
                graph_model, graph_tokenizer,
                feature_mappings)
    
    def generate_baseline(self, model, tokenizer, prompt: str,
                         max_new_tokens: int = 384) -> str:
        """Generate completion using baseline model"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def generate_graph(self, model, tokenizer, prompt: str,
                      ground_truth_code: str,
                      feature_mappings: Dict,
                      use_graphs: bool = True,
                      max_new_tokens: int = 384) -> str:
        """
        Generate completion using graph model
        
        Args:
            use_graphs: If True, use graphs. If False, run without graphs (ablation)
        """
        
        # Build graphs from ground truth code (if using graphs)
        if use_graphs:
            ast_batch, cfg_batch, dfg_batch = self.build_graphs_from_code(
                ground_truth_code, feature_mappings
            )
        else:
            ast_batch, cfg_batch, dfg_batch = None, None, None
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            generated_ids = generate_with_layerwise_graphs(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                ast_batch=ast_batch,
                cfg_batch=cfg_batch,
                dfg_batch=dfg_batch,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.95,
                do_sample=True
            )
        
        full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response = full_response[len(prompt):] if len(full_response) > len(prompt) else full_response
        
        return response.strip()
    
    def build_graphs_from_code(self, code: str, feature_mappings: Dict):
        """Build graphs from ground truth code"""
        try:
            # Build AST
            ast_builder = ASTGraphBuilder()
            ast_nodes, ast_edges = ast_builder.build(code)
            
            if len(ast_nodes) > 0:
                ast_type2id = feature_mappings['ast_type2id']
                ast_num_classes = feature_mappings['ast_num_classes']
                target_dim = feature_mappings['target_dim']
                
                indices = [ast_type2id.get(typ, 0) for typ in ast_nodes]
                x_ast = torch.nn.functional.one_hot(
                    torch.tensor(indices),
                    num_classes=ast_num_classes
                ).float()
                
                if x_ast.shape[1] < target_dim:
                    pad = torch.zeros(x_ast.shape[0], target_dim - x_ast.shape[1])
                    x_ast = torch.cat([x_ast, pad], dim=1)
                elif x_ast.shape[1] > target_dim:
                    x_ast = x_ast[:, :target_dim]
                
                edge_index = torch.tensor(ast_edges, dtype=torch.long).t().contiguous() if ast_edges else torch.empty((2, 0), dtype=torch.long)
                ast_batch = Data(x=x_ast, edge_index=edge_index).to(self.device)
            else:
                ast_batch = None
            
            # Build CFG
            try:
                cfg_batch = cfg_to_pyg_data(code).to(self.device)
            except:
                cfg_batch = None
            
            # Build DFG
            dfg_builder = DFGBuilder()
            dfg_nodes, dfg_edges = dfg_builder.build(code)
            
            if len(dfg_nodes) > 0:
                dfg_type2id = feature_mappings['dfg_type2id']
                dfg_num_classes = feature_mappings['dfg_num_classes']
                
                indices = [dfg_type2id.get(typ, 0) for typ in dfg_nodes]
                x_dfg = torch.nn.functional.one_hot(
                    torch.tensor(indices),
                    num_classes=dfg_num_classes
                ).float()
                
                if x_dfg.shape[1] < target_dim:
                    pad = torch.zeros(x_dfg.shape[0], target_dim - x_dfg.shape[1])
                    x_dfg = torch.cat([x_dfg, pad], dim=1)
                elif x_dfg.shape[1] > target_dim:
                    x_dfg = x_dfg[:, :target_dim]
                
                edge_index = torch.tensor(dfg_edges, dtype=torch.long).t().contiguous() if dfg_edges else torch.empty((2, 0), dtype=torch.long)
                dfg_batch = Data(x=x_dfg, edge_index=edge_index).to(self.device)
            else:
                dfg_batch = None
            
            return ast_batch, cfg_batch, dfg_batch
            
        except Exception as e:
            print(f"Warning: Could not build graphs from code: {e}")
            return None, None, None
    
    # def run_benchmark(self):
    #     """Run comprehensive InstructCoder benchmark"""
    #     print("="*70)
    #     print("INSTRUCTCODER BENCHMARK WITH GEMINI TEST GENERATION")
    #     print("Testing 6 variants: 3 models x 2 extraction methods")
    #     print("="*70)
        
    #     # Load dataset
    #     dataset = self.load_instructcoder_dataset()
        
    #     # Load models
    #     (baseline_model, baseline_tokenizer,
    #      graph_model, graph_tokenizer,
    #      feature_mappings) = self.load_models()
        
    #     start_time = time.time()
        
    #     for i, sample in enumerate(tqdm(dataset, desc="Evaluating InstructCoder")):
    #         try:
    #             instruction = sample['instruction']
    #             ground_truth = sample['output']
                
    #             print(f"\n{'='*70}")
    #             print(f"Problem {i}: {instruction[:100]}...")
    #             print(f"{'='*70}")
                
    #             # Generate test cases using Gemini
    #             print("🤖 Generating test cases with Gemini...")
    #             test_code, test_metadata = self.test_generator.generate_test_cases(
    #                 instruction,
    #                 ground_truth,
    #                 num_tests=self.num_tests_per_problem
    #             )
                
    #             if not test_code:
    #                 print(f"⚠️  Could not generate test cases for problem {i}, skipping...")
    #                 continue
                
    #             # Validate test code
    #             is_valid, error_msg = self.test_generator.validate_test_code(test_code, ground_truth)
    #             if not is_valid:
    #                 print(f"⚠️  Generated test code failed validation: {error_msg}")
    #                 print("Skipping this problem...")
    #                 continue
                
    #             print(f"✓ Generated and validated {len(test_metadata)} test cases")
                
    #             # Save generated test cases
    #             if self.save_code_files:
    #                 test_case_path = os.path.join(
    #                     self.code_output_dir, 
    #                     "generated_test_cases",
    #                     f"problem_{i:03d}_tests.py"
    #                 )
    #                 with open(test_case_path, 'w', encoding='utf-8') as f:
    #                     f.write("# " + "="*68 + "\n")
    #                     f.write(f"# Problem {i}: {instruction[:60]}...\n")
    #                     f.write("# " + "="*68 + "\n")
    #                     f.write("# GEMINI-GENERATED TEST CASES\n")
    #                     f.write("# " + "="*68 + "\n\n")
    #                     f.write(ground_truth)
    #                     f.write("\n\n")
    #                     f.write(test_code)
                    
    #                 # Save ground truth
    #                 gt_path = os.path.join(self.code_output_dir, "ground_truth", f"problem_{i:03d}_gt.py")
    #                 with open(gt_path, 'w', encoding='utf-8') as f:
    #                     f.write(ground_truth)
                
    #             # Store test case info
    #             self.results['test_cases'].append({
    #                 'problem_idx': i,
    #                 'instruction': instruction,
    #                 'test_code': test_code,
    #                 'num_tests': len(test_metadata),
    #                 'test_metadata': test_metadata
    #             })
                
    #             # Create prompt
    #             prompt = self.create_prompt_from_instruction(instruction)
                
    #             # 1. Generate BASELINE completion
    #             print("📝 Generating baseline completion...")
    #             baseline_start = time.time()
    #             baseline_completion = self.generate_baseline(
    #                 baseline_model, baseline_tokenizer, prompt
    #             )
    #             baseline_time = time.time() - baseline_start
                
    #             # 2. Generate GRAPH WITH GRAPHS completion
    #             print("📊 Generating graph (WITH graphs) completion...")
    #             graph_with_start = time.time()
    #             graph_with_completion = self.generate_graph(
    #                 graph_model, graph_tokenizer, prompt,
    #                 ground_truth, feature_mappings,
    #                 use_graphs=True
    #             )
    #             graph_with_time = time.time() - graph_with_start
                
    #             # 3. Generate GRAPH WITHOUT GRAPHS completion (ablation)
    #             print("🔍 Generating graph (NO graphs) completion...")
    #             graph_no_start = time.time()
    #             graph_no_completion = self.generate_graph(
    #                 graph_model, graph_tokenizer, prompt,
    #                 ground_truth, feature_mappings,
    #                 use_graphs=False
    #             )
    #             graph_no_time = time.time() - graph_no_start
                
    #             # Extract code - SIMPLE (concatenate as-is)
    #             baseline_code_simple = self.code_extractor.simple_concatenate(
    #                 baseline_completion, ""
    #             )
    #             graph_with_code_simple = self.code_extractor.simple_concatenate(
    #                 graph_with_completion, ""
    #             )
    #             graph_no_code_simple = self.code_extractor.simple_concatenate(
    #                 graph_no_completion, ""
    #             )
                
    #             # Extract code - CLEAN (stop at test markers)
    #             baseline_code_clean, baseline_had_tests = self.code_extractor.extract_until_test_markers(
    #                 baseline_completion, ""
    #             )
    #             graph_with_code_clean, graph_with_had_tests = self.code_extractor.extract_until_test_markers(
    #                 graph_with_completion, ""
    #             )
    #             graph_no_code_clean, graph_no_had_tests = self.code_extractor.extract_until_test_markers(
    #                 graph_no_completion, ""
    #             )
                
    #             print(f"\n📏 Code lengths (simple extraction):")
    #             print(f"  Baseline:          {len(baseline_code_simple)} chars, {len(baseline_code_simple.splitlines())} lines")
    #             print(f"  Graph (w/ graphs): {len(graph_with_code_simple)} chars, {len(graph_with_code_simple.splitlines())} lines")
    #             print(f"  Graph (no graphs): {len(graph_no_code_simple)} chars, {len(graph_no_code_simple.splitlines())} lines")
                
    #             # Check syntax for all 6 versions
    #             baseline_simple_syntax = self.evaluator.syntax_check(baseline_code_simple)
    #             baseline_clean_syntax = self.evaluator.syntax_check(baseline_code_clean)
    #             graph_with_simple_syntax = self.evaluator.syntax_check(graph_with_code_simple)
    #             graph_with_clean_syntax = self.evaluator.syntax_check(graph_with_code_clean)
    #             graph_no_simple_syntax = self.evaluator.syntax_check(graph_no_code_simple)
    #             graph_no_clean_syntax = self.evaluator.syntax_check(graph_no_code_clean)
                
    #             # Prepare test
    #             test_list = [{'test': test_code}]
                
    #             # Evaluate all 6 versions
    #             print("🧪 Running tests on all 6 variants...")
    #             baseline_simple_results = self.evaluator.evaluate_functional_correctness(
    #                 baseline_code_simple, test_list
    #             )
    #             baseline_clean_results = self.evaluator.evaluate_functional_correctness(
    #                 baseline_code_clean, test_list
    #             )
    #             graph_with_simple_results = self.evaluator.evaluate_functional_correctness(
    #                 graph_with_code_simple, test_list
    #             )
    #             graph_with_clean_results = self.evaluator.evaluate_functional_correctness(
    #                 graph_with_code_clean, test_list
    #             )
    #             graph_no_simple_results = self.evaluator.evaluate_functional_correctness(
    #                 graph_no_code_simple, test_list
    #             )
    #             graph_no_clean_results = self.evaluator.evaluate_functional_correctness(
    #                 graph_no_code_clean, test_list
    #             )
                
    #             print(f"\n📊 Results:")
    #             print(f"  Baseline Simple:          Syntax: {baseline_simple_syntax}, Pass: {baseline_simple_results['passed']}/{len(test_metadata)}")
    #             print(f"  Baseline Clean:           Syntax: {baseline_clean_syntax}, Pass: {baseline_clean_results['passed']}/{len(test_metadata)}")
    #             print(f"  Graph(w/ graphs) Simple:  Syntax: {graph_with_simple_syntax}, Pass: {graph_with_simple_results['passed']}/{len(test_metadata)}")
    #             print(f"  Graph(w/ graphs) Clean:   Syntax: {graph_with_clean_syntax}, Pass: {graph_with_clean_results['passed']}/{len(test_metadata)}")
    #             print(f"  Graph(no graphs) Simple:  Syntax: {graph_no_simple_syntax}, Pass: {graph_no_simple_results['passed']}/{len(test_metadata)}")
    #             print(f"  Graph(no graphs) Clean:   Syntax: {graph_no_clean_syntax}, Pass: {graph_no_clean_results['passed']}/{len(test_metadata)}")
                
    #             # Calculate similarities
    #             sim_baseline_simple_vs_graph_with_simple = self.calculate_code_similarity(baseline_code_simple, graph_with_code_simple)
    #             sim_baseline_clean_vs_graph_with_clean = self.calculate_code_similarity(baseline_code_clean, graph_with_code_clean)
    #             sim_graph_with_simple_vs_graph_no_simple = self.calculate_code_similarity(graph_with_code_simple, graph_no_code_simple)
                
    #             print(f"\n🔄 Code similarities:")
    #             print(f"  Baseline(simple) vs Graph-with(simple): {sim_baseline_simple_vs_graph_with_simple:.3f}")
    #             print(f"  Baseline(clean) vs Graph-with(clean):   {sim_baseline_clean_vs_graph_with_clean:.3f}")
    #             print(f"  Graph-with(simple) vs Graph-no(simple): {sim_graph_with_simple_vs_graph_no_simple:.3f}")
                
    #             # Save all 6 versions
    #             if self.save_code_files:
    #                 versions = [
    #                     ("baseline_simple", baseline_code_simple, baseline_simple_results, baseline_simple_syntax, baseline_time),
    #                     ("baseline_clean", baseline_code_clean, baseline_clean_results, baseline_clean_syntax, baseline_time),
    #                     ("graph_with_graphs_simple", graph_with_code_simple, graph_with_simple_results, graph_with_simple_syntax, graph_with_time),
    #                     ("graph_with_graphs_clean", graph_with_code_clean, graph_with_clean_results, graph_with_clean_syntax, graph_with_time),
    #                     ("graph_no_graphs_simple", graph_no_code_simple, graph_no_simple_results, graph_no_simple_syntax, graph_no_time),
    #                     ("graph_no_graphs_clean", graph_no_code_clean, graph_no_clean_results, graph_no_clean_syntax, graph_no_time),
    #                 ]
                    
    #                 for model_type, code, results, syntax, inf_time in versions:
    #                     self.save_generated_code_file(
    #                         code=code,
    #                         test=test_code,
    #                         task_id=f"InstructCoder_{i}",
    #                         instruction=instruction,
    #                         model_type=model_type,
    #                         problem_idx=i,
    #                         test_results=results,
    #                         syntax_valid=syntax,
    #                         inference_time=inf_time
    #                     )
                
    #             # Store results for all 6 versions
    #             self.results['baseline_simple'].append({
    #                 'task_id': f"InstructCoder_{i}",
    #                 'instruction': instruction,
    #                 'syntax_valid': baseline_simple_syntax,
    #                 'pass_rate': baseline_simple_results['pass_rate'],
    #                 'passed': baseline_simple_results.get('passed', 0),
    #                 'failed': baseline_simple_results.get('failed', 0),
    #                 'num_tests': len(test_metadata),
    #                 'inference_time': baseline_time,
    #                 'code': baseline_code_simple,
    #                 'code_length': len(baseline_code_simple),
    #                 'num_lines': len(baseline_code_simple.splitlines()),
    #             })
                
    #             self.results['baseline_clean'].append({
    #                 'task_id': f"InstructCoder_{i}",
    #                 'instruction': instruction,
    #                 'syntax_valid': baseline_clean_syntax,
    #                 'pass_rate': baseline_clean_results['pass_rate'],
    #                 'passed': baseline_clean_results.get('passed', 0),
    #                 'failed': baseline_clean_results.get('failed', 0),
    #                 'num_tests': len(test_metadata),
    #                 'inference_time': baseline_time,
    #                 'code': baseline_code_clean,
    #                 'code_length': len(baseline_code_clean),
    #                 'num_lines': len(baseline_code_clean.splitlines()),
    #                 'had_test_code': baseline_had_tests
    #             })
                
    #             self.results['graph_with_graphs_simple'].append({
    #                 'task_id': f"InstructCoder_{i}",
    #                 'instruction': instruction,
    #                 'syntax_valid': graph_with_simple_syntax,
    #                 'pass_rate': graph_with_simple_results['pass_rate'],
    #                 'passed': graph_with_simple_results.get('passed', 0),
    #                 'failed': graph_with_simple_results.get('failed', 0),
    #                 'num_tests': len(test_metadata),
    #                 'inference_time': graph_with_time,
    #                 'code': graph_with_code_simple,
    #                 'code_length': len(graph_with_code_simple),
    #                 'num_lines': len(graph_with_code_simple.splitlines()),
    #             })
                
    #             self.results['graph_with_graphs_clean'].append({
    #                 'task_id': f"InstructCoder_{i}",
    #                 'instruction': instruction,
    #                 'syntax_valid': graph_with_clean_syntax,
    #                 'pass_rate': graph_with_clean_results['pass_rate'],
    #                 'passed': graph_with_clean_results.get('passed', 0),
    #                 'failed': graph_with_clean_results.get('failed', 0),
    #                 'num_tests': len(test_metadata),
    #                 'inference_time': graph_with_time,
    #                 'code': graph_with_code_clean,
    #                 'code_length': len(graph_with_code_clean),
    #                 'num_lines': len(graph_with_code_clean.splitlines()),
    #                 'had_test_code': graph_with_had_tests
    #             })
                
    #             self.results['graph_no_graphs_simple'].append({
    #                 'task_id': f"InstructCoder_{i}",
    #                 'instruction': instruction,
    #                 'syntax_valid': graph_no_simple_syntax,
    #                 'pass_rate': graph_no_simple_results['pass_rate'],
    #                 'passed': graph_no_simple_results.get('passed', 0),
    #                 'failed': graph_no_simple_results.get('failed', 0),
    #                 'num_tests': len(test_metadata),
    #                 'inference_time': graph_no_time,
    #                 'code': graph_no_code_simple,
    #                 'code_length': len(graph_no_code_simple),
    #                 'num_lines': len(graph_no_code_simple.splitlines()),
    #             })
                
    #             self.results['graph_no_graphs_clean'].append({
    #                 'task_id': f"InstructCoder_{i}",
    #                 'instruction': instruction,
    #                 'syntax_valid': graph_no_clean_syntax,
    #                 'pass_rate': graph_no_clean_results['pass_rate'],
    #                 'passed': graph_no_clean_results.get('passed', 0),
    #                 'failed': graph_no_clean_results.get('failed', 0),
    #                 'num_tests': len(test_metadata),
    #                 'inference_time': graph_no_time,
    #                 'code': graph_no_code_clean,
    #                 'code_length': len(graph_no_code_clean),
    #                 'num_lines': len(graph_no_code_clean.splitlines()),
    #                 'had_test_code': graph_no_had_tests
    #             })
                
    #             self.results['metadata'].append({
    #                 'problem_idx': i,
    #                 'task_id': f"InstructCoder_{i}",
    #                 'instruction': instruction,
    #                 'ground_truth': ground_truth,
    #                 'num_generated_tests': len(test_metadata),
    #                 'baseline_had_tests': baseline_had_tests,
    #                 'graph_with_had_tests': graph_with_had_tests,
    #                 'graph_no_had_tests': graph_no_had_tests,
    #                 'similarity_baseline_simple_vs_graph_with_simple': sim_baseline_simple_vs_graph_with_simple,
    #                 'similarity_baseline_clean_vs_graph_with_clean': sim_baseline_clean_vs_graph_with_clean,
    #                 'similarity_graph_with_simple_vs_graph_no_simple': sim_graph_with_simple_vs_graph_no_simple,
    #             })
                
    #             # Rate limiting for Gemini API
    #             time.sleep(1)
                
    #         except Exception as e:
    #             print(f"❌ Error on problem {i}: {e}")
    #             import traceback
    #             traceback.print_exc()
    #             continue
        
    #     total_time = time.time() - start_time
    #     print(f"\n{'='*70}")
    #     print(f"Benchmark completed in {total_time:.2f} seconds")
    #     print(f"{'='*70}")
        
    #     self.calculate_final_metrics()
    #     self.save_results()
    def run_benchmark(self):
        """Run comprehensive InstructCoder benchmark"""
        print("="*70)
        print("INSTRUCTCODER BENCHMARK WITH GEMINI TEST GENERATION")
        print("Testing 6 variants: 3 models x 2 extraction methods")
        print("="*70)
        
        # Load dataset
        dataset = self.load_instructcoder_dataset()
        
        # Load models
        (baseline_model, baseline_tokenizer,
        graph_model, graph_tokenizer,
        feature_mappings) = self.load_models()
        
        start_time = time.time()
        
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating InstructCoder")):
            try:
                instruction = sample['instruction']
                ground_truth = sample['output']
                # original_prompt = sample['prompt']
                
                print(f"\n{'='*70}")
                print(f"Problem {i}: {instruction[:100]}...")
                print(f"{'='*70}")
                
                # Generate test cases using Gemini
                print("🤖 Generating test cases with Gemini...")
                test_code, test_metadata = self.test_generator.generate_test_cases(
                    instruction,
                    ground_truth,
                    num_tests=self.num_tests_per_problem
                )
                
                if not test_code:
                    print(f"⚠️  Could not generate test cases for problem {i}, skipping...")
                    continue
                
                # Validate test code
                is_valid, error_msg = self.test_generator.validate_test_code(test_code, ground_truth)
                if not is_valid:
                    print(f"⚠️  Generated test code failed validation: {error_msg}")
                    print("Skipping this problem...")
                    continue
                
                print(f"✓ Generated and validated {len(test_metadata)} test cases")
                
                # Save generated test cases
                if self.save_code_files:
                    test_case_path = os.path.join(
                        self.code_output_dir, 
                        "generated_test_cases",
                        f"problem_{i:03d}_tests.py"
                    )
                    with open(test_case_path, 'w', encoding='utf-8') as f:
                        f.write("# " + "="*68 + "\n")
                        f.write(f"# Problem {i}: {instruction[:60]}...\n")
                        f.write("# " + "="*68 + "\n")
                        f.write("# GEMINI-GENERATED TEST CASES\n")
                        f.write(f"# Number of test cases: {len(test_metadata)}\n")
                        f.write("# " + "="*68 + "\n\n")
                        f.write(ground_truth)
                        f.write("\n\n")
                        f.write(test_code)
                    
                    # Save ground truth
                    gt_path = os.path.join(self.code_output_dir, "ground_truth", f"problem_{i:03d}_gt.py")
                    with open(gt_path, 'w', encoding='utf-8') as f:
                        f.write(ground_truth)
                
                # Store test case info
                self.results['test_cases'].append({
                    'problem_idx': i,
                    'instruction': instruction,
                    'test_code': test_code,
                    'num_tests': len(test_metadata),
                    'test_metadata': test_metadata
                })
                
                # Create prompt
                prompt = self.create_prompt_from_instruction(instruction)
                
                # 1. Generate BASELINE completion
                print("📝 Generating baseline completion...")
                baseline_start = time.time()
                baseline_completion = self.generate_baseline(
                    baseline_model, baseline_tokenizer, prompt
                )
                baseline_time = time.time() - baseline_start
                
                # 2. Generate GRAPH WITH GRAPHS completion
                print("📊 Generating graph (WITH graphs) completion...")
                graph_with_start = time.time()
                graph_with_completion = self.generate_graph(
                    graph_model, graph_tokenizer, prompt,
                    ground_truth, feature_mappings,
                    use_graphs=True
                )
                graph_with_time = time.time() - graph_with_start
                
                # 3. Generate GRAPH WITHOUT GRAPHS completion (ablation)
                print("🔍 Generating graph (NO graphs) completion...")
                graph_no_start = time.time()
                graph_no_completion = self.generate_graph(
                    graph_model, graph_tokenizer, prompt,
                    ground_truth, feature_mappings,
                    use_graphs=False
                )
                graph_no_time = time.time() - graph_no_start
                
                # Extract code - SIMPLE (concatenate as-is)
                baseline_code_simple = self.code_extractor.simple_concatenate(
                    baseline_completion, ""

                )
                graph_with_code_simple = self.code_extractor.simple_concatenate(
                    graph_with_completion, ""
                )
                graph_no_code_simple = self.code_extractor.simple_concatenate(
                    graph_no_completion, ""
                )
                
                # Extract code - CLEAN (stop at test markers)
                baseline_code_clean, baseline_had_tests = self.code_extractor.extract_until_test_markers(
                    baseline_completion, ""
                )
                graph_with_code_clean, graph_with_had_tests = self.code_extractor.extract_until_test_markers(
                    graph_with_completion, ""
                )
                graph_no_code_clean, graph_no_had_tests = self.code_extractor.extract_until_test_markers(
                    graph_no_completion, ""
                )
                
                print(f"\n📏 Code lengths (simple extraction):")
                print(f"  Baseline:          {len(baseline_code_simple)} chars, {len(baseline_code_simple.splitlines())} lines")
                print(f"  Graph (w/ graphs): {len(graph_with_code_simple)} chars, {len(graph_with_code_simple.splitlines())} lines")
                print(f"  Graph (no graphs): {len(graph_no_code_simple)} chars, {len(graph_no_code_simple.splitlines())} lines")
                
                # Check syntax for all 6 versions
                baseline_simple_syntax = self.evaluator.syntax_check(baseline_code_simple)
                baseline_clean_syntax = self.evaluator.syntax_check(baseline_code_clean)
                graph_with_simple_syntax = self.evaluator.syntax_check(graph_with_code_simple)
                graph_with_clean_syntax = self.evaluator.syntax_check(graph_with_code_clean)
                graph_no_simple_syntax = self.evaluator.syntax_check(graph_no_code_simple)
                graph_no_clean_syntax = self.evaluator.syntax_check(graph_no_code_clean)
                
                # Measure code execution time (how fast the code itself runs)
                print("⏱️  Measuring code execution times...")
                baseline_simple_exec = self.evaluator.measure_code_execution_time(baseline_code_simple)
                baseline_clean_exec = self.evaluator.measure_code_execution_time(baseline_code_clean)
                graph_with_simple_exec = self.evaluator.measure_code_execution_time(graph_with_code_simple)
                graph_with_clean_exec = self.evaluator.measure_code_execution_time(graph_with_code_clean)
                graph_no_simple_exec = self.evaluator.measure_code_execution_time(graph_no_code_simple)
                graph_no_clean_exec = self.evaluator.measure_code_execution_time(graph_no_code_clean)
                
                # Prepare test
                test_list = [{'test': test_code}]
                
                # Evaluate all 6 versions
                print("🧪 Running tests on all 6 variants...")
                baseline_simple_results = self.evaluator.evaluate_functional_correctness(
                    baseline_code_simple, test_list
                )
                baseline_clean_results = self.evaluator.evaluate_functional_correctness(
                    baseline_code_clean, test_list
                )
                graph_with_simple_results = self.evaluator.evaluate_functional_correctness(
                    graph_with_code_simple, test_list
                )
                graph_with_clean_results = self.evaluator.evaluate_functional_correctness(
                    graph_with_code_clean, test_list
                )
                graph_no_simple_results = self.evaluator.evaluate_functional_correctness(
                    graph_no_code_simple, test_list
                )
                graph_no_clean_results = self.evaluator.evaluate_functional_correctness(
                    graph_no_code_clean, test_list
                )
                
                # print(f"\n📊 Results:")
                # print(f"  Baseline Simple:          Syntax: {baseline_simple_syntax}, Pass: {baseline_simple_results['passed']}/{len(test_metadata)}, Exec: {baseline_simple_exec['avg_time']*1000:.2f}ms")
                # print(f"  Baseline Clean:           Syntax: {baseline_clean_syntax}, Pass: {baseline_clean_results['passed']}/{len(test_metadata)}, Exec: {baseline_clean_exec['avg_time']*1000:.2f}ms")
                # print(f"  Graph(w/ graphs) Simple:  Syntax: {graph_with_simple_syntax}, Pass: {graph_with_simple_results['passed']}/{len(test_metadata)}, Exec: {graph_with_simple_exec['avg_time']*1000:.2f}ms")
                # print(f"  Graph(w/ graphs) Clean:   Syntax: {graph_with_clean_syntax}, Pass: {graph_with_clean_results['passed']}/{len(test_metadata)}, Exec: {graph_with_clean_exec['avg_time']*1000:.2f}ms")
                # print(f"  Graph(no graphs) Simple:  Syntax: {graph_no_simple_syntax}, Pass: {graph_no_simple_results['passed']}/{len(test_metadata)}, Exec: {graph_no_simple_exec['avg_time']*1000:.2f}ms")
                # print(f"  Graph(no graphs) Clean:   Syntax: {graph_no_clean_syntax}, Pass: {graph_no_clean_results['passed']}/{len(test_metadata)}, Exec: {graph_no_clean_exec['avg_time']*1000:.2f}ms")
                # In the run_benchmark method, update the print statement:

                print(f"\n📊 Results:")
                print(f"  Baseline Simple:          Syntax: {baseline_simple_syntax}, Pass: {baseline_simple_results['passed']}/{len(test_metadata)}, Exec: {baseline_simple_exec['avg_time']*1000:.2f}ms {'✓' if baseline_simple_exec['success'] else '✗'}")
                print(f"  Baseline Clean:           Syntax: {baseline_clean_syntax}, Pass: {baseline_clean_results['passed']}/{len(test_metadata)}, Exec: {baseline_clean_exec['avg_time']*1000:.2f}ms {'✓' if baseline_clean_exec['success'] else '✗'}")
                print(f"  Graph(w/ graphs) Simple:  Syntax: {graph_with_simple_syntax}, Pass: {graph_with_simple_results['passed']}/{len(test_metadata)}, Exec: {graph_with_simple_exec['avg_time']*1000:.2f}ms {'✓' if graph_with_simple_exec['success'] else '✗'}")
                print(f"  Graph(w/ graphs) Clean:   Syntax: {graph_with_clean_syntax}, Pass: {graph_with_clean_results['passed']}/{len(test_metadata)}, Exec: {graph_with_clean_exec['avg_time']*1000:.2f}ms {'✓' if graph_with_clean_exec['success'] else '✗'}")
                print(f"  Graph(no graphs) Simple:  Syntax: {graph_no_simple_syntax}, Pass: {graph_no_simple_results['passed']}/{len(test_metadata)}, Exec: {graph_no_simple_exec['avg_time']*1000:.2f}ms {'✓' if graph_no_simple_exec['success'] else '✗'}")
                print(f"  Graph(no graphs) Clean:   Syntax: {graph_no_clean_syntax}, Pass: {graph_no_clean_results['passed']}/{len(test_metadata)}, Exec: {graph_no_clean_exec['avg_time']*1000:.2f}ms {'✓' if graph_no_clean_exec['success'] else '✗'}")

                # Add execution errors if any
                if not baseline_simple_exec['success']:
                    print(f"    Baseline Simple exec error: {baseline_simple_exec.get('error', 'Unknown')[:50]}")
                if not baseline_clean_exec['success']:
                    print(f"    Baseline Clean exec error: {baseline_clean_exec.get('error', 'Unknown')[:50]}")
                
                # Calculate similarities
                sim_baseline_simple_vs_graph_with_simple = self.calculate_code_similarity(baseline_code_simple, graph_with_code_simple)
                sim_baseline_clean_vs_graph_with_clean = self.calculate_code_similarity(baseline_code_clean, graph_with_code_clean)
                sim_graph_with_simple_vs_graph_no_simple = self.calculate_code_similarity(graph_with_code_simple, graph_no_code_simple)
                
                print(f"\n🔄 Code similarities:")
                print(f"  Baseline(simple) vs Graph-with(simple): {sim_baseline_simple_vs_graph_with_simple:.3f}")
                print(f"  Baseline(clean) vs Graph-with(clean):   {sim_baseline_clean_vs_graph_with_clean:.3f}")
                print(f"  Graph-with(simple) vs Graph-no(simple): {sim_graph_with_simple_vs_graph_no_simple:.3f}")
                
                # Save all 6 versions
                if self.save_code_files:
                    versions = [
                        ("baseline_simple", baseline_code_simple, baseline_simple_results, baseline_simple_syntax, baseline_time, baseline_simple_exec),
                        ("baseline_clean", baseline_code_clean, baseline_clean_results, baseline_clean_syntax, baseline_time, baseline_clean_exec),
                        ("graph_with_graphs_simple", graph_with_code_simple, graph_with_simple_results, graph_with_simple_syntax, graph_with_time, graph_with_simple_exec),
                        ("graph_with_graphs_clean", graph_with_code_clean, graph_with_clean_results, graph_with_clean_syntax, graph_with_time, graph_with_clean_exec),
                        ("graph_no_graphs_simple", graph_no_code_simple, graph_no_simple_results, graph_no_simple_syntax, graph_no_time, graph_no_simple_exec),
                        ("graph_no_graphs_clean", graph_no_code_clean, graph_no_clean_results, graph_no_clean_syntax, graph_no_time, graph_no_clean_exec),
                    ]
                    
                    for model_type, code, results, syntax, inf_time, exec_timing in versions:
                        self.save_generated_code_file(
                            code=code,
                            test=test_code,
                            task_id=f"InstructCoder_{i}",
                            instruction=instruction,
                            model_type=model_type,
                            problem_idx=i,
                            test_results=results,
                            syntax_valid=syntax,
                            inference_time=inf_time,
                            execution_timing=exec_timing
                        )
                
                # Store results for all 6 versions (including execution timing)
                self.results['baseline_simple'].append({
                    'task_id': f"InstructCoder_{i}",
                    'instruction': instruction,
                    'syntax_valid': baseline_simple_syntax,
                    'pass_rate': baseline_simple_results['pass_rate'],
                    'passed': baseline_simple_results.get('passed', 0),
                    'failed': baseline_simple_results.get('failed', 0),
                    'num_tests': len(test_metadata),
                    'inference_time': baseline_time,
                    'test_execution_time': baseline_simple_results['execution_time'],
                    'code_execution_time': baseline_simple_exec['avg_time'],
                    'code_exec_details': baseline_simple_exec,
                    'code': baseline_code_simple,
                    'code_length': len(baseline_code_simple),
                    'num_lines': len(baseline_code_simple.splitlines()),
                })
                
                self.results['baseline_clean'].append({
                    'task_id': f"InstructCoder_{i}",
                    'instruction': instruction,
                    'syntax_valid': baseline_clean_syntax,
                    'pass_rate': baseline_clean_results['pass_rate'],
                    'passed': baseline_clean_results.get('passed', 0),
                    'failed': baseline_clean_results.get('failed', 0),
                    'num_tests': len(test_metadata),
                    'inference_time': baseline_time,
                    'test_execution_time': baseline_clean_results['execution_time'],
                    'code_execution_time': baseline_clean_exec['avg_time'],
                    'code_exec_details': baseline_clean_exec,
                    'code': baseline_code_clean,
                    'code_length': len(baseline_code_clean),
                    'num_lines': len(baseline_code_clean.splitlines()),
                    'had_test_code': baseline_had_tests
                })
                
                self.results['graph_with_graphs_simple'].append({
                    'task_id': f"InstructCoder_{i}",
                    'instruction': instruction,
                    'syntax_valid': graph_with_simple_syntax,
                    'pass_rate': graph_with_simple_results['pass_rate'],
                    'passed': graph_with_simple_results.get('passed', 0),
                    'failed': graph_with_simple_results.get('failed', 0),
                    'num_tests': len(test_metadata),
                    'inference_time': graph_with_time,
                    'test_execution_time': graph_with_simple_results['execution_time'],
                    'code_execution_time': graph_with_simple_exec['avg_time'],
                    'code_exec_details': graph_with_simple_exec,
                    'code': graph_with_code_simple,
                    'code_length': len(graph_with_code_simple),
                    'num_lines': len(graph_with_code_simple.splitlines()),
                })
                
                self.results['graph_with_graphs_clean'].append({
                    'task_id': f"InstructCoder_{i}",
                    'instruction': instruction,
                    'syntax_valid': graph_with_clean_syntax,
                    'pass_rate': graph_with_clean_results['pass_rate'],
                    'passed': graph_with_clean_results.get('passed', 0),
                    'failed': graph_with_clean_results.get('failed', 0),
                    'num_tests': len(test_metadata),
                    'inference_time': graph_with_time,
                    'test_execution_time': graph_with_clean_results['execution_time'],
                    'code_execution_time': graph_with_clean_exec['avg_time'],
                    'code_exec_details': graph_with_clean_exec,
                    'code': graph_with_code_clean,
                    'code_length': len(graph_with_code_clean),
                    'num_lines': len(graph_with_code_clean.splitlines()),
                    'had_test_code': graph_with_had_tests
                })
                
                self.results['graph_no_graphs_simple'].append({
                    'task_id': f"InstructCoder_{i}",
                    'instruction': instruction,
                    'syntax_valid': graph_no_simple_syntax,
                    'pass_rate': graph_no_simple_results['pass_rate'],
                    'passed': graph_no_simple_results.get('passed', 0),
                    'failed': graph_no_simple_results.get('failed', 0),
                    'num_tests': len(test_metadata),
                    'inference_time': graph_no_time,
                    'test_execution_time': graph_no_simple_results['execution_time'],
                    'code_execution_time': graph_no_simple_exec['avg_time'],
                    'code_exec_details': graph_no_simple_exec,
                    'code': graph_no_code_simple,
                    'code_length': len(graph_no_code_simple),
                    'num_lines': len(graph_no_code_simple.splitlines()),
                })
                
                self.results['graph_no_graphs_clean'].append({
                    'task_id': f"InstructCoder_{i}",
                    'instruction': instruction,
                    'syntax_valid': graph_no_clean_syntax,
                    'pass_rate': graph_no_clean_results['pass_rate'],
                    'passed': graph_no_clean_results.get('passed', 0),
                    'failed': graph_no_clean_results.get('failed', 0),
                    'num_tests': len(test_metadata),
                    'inference_time': graph_no_time,
                    'test_execution_time': graph_no_clean_results['execution_time'],
                    'code_execution_time': graph_no_clean_exec['avg_time'],
                    'code_exec_details': graph_no_clean_exec,
                    'code': graph_no_code_clean,
                    'code_length': len(graph_no_code_clean),
                    'num_lines': len(graph_no_code_clean.splitlines()),
                    'had_test_code': graph_no_had_tests
                })
                
                self.results['metadata'].append({
                    'problem_idx': i,
                    'task_id': f"InstructCoder_{i}",
                    'instruction': instruction,
                    'ground_truth': ground_truth,
                    'num_generated_tests': len(test_metadata),
                    'baseline_had_tests': baseline_had_tests,
                    'graph_with_had_tests': graph_with_had_tests,
                    'graph_no_had_tests': graph_no_had_tests,
                    'similarity_baseline_simple_vs_graph_with_simple': sim_baseline_simple_vs_graph_with_simple,
                    'similarity_baseline_clean_vs_graph_with_clean': sim_baseline_clean_vs_graph_with_clean,
                    'similarity_graph_with_simple_vs_graph_no_simple': sim_graph_with_simple_vs_graph_no_simple,
                })
                
                # Rate limiting for Gemini API
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error on problem {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Benchmark completed in {total_time:.2f} seconds")
        print(f"{'='*70}")
        
        self.calculate_final_metrics()
        self.save_results()

    def calculate_code_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, code1, code2).ratio()
    
    # def save_generated_code_file(self, code: str, test: str, task_id: str,
    #                             instruction: str, model_type: str, problem_idx: int,
    #                             test_results: Dict = None,
    #                             syntax_valid: bool = True,
    #                             inference_time: float = 0.0):
    #     """Save generated code to file"""
    #     if not self.save_code_files:
    #         return
        
    #     try:
    #         safe_task_id = task_id.replace("/", "_").replace(" ", "_")
    #         model_dir = os.path.join(self.code_output_dir, model_type)
            
    #         code_filename = f"problem_{problem_idx:03d}_{safe_task_id}.py"
    #         code_path = os.path.join(model_dir, code_filename)
            
    #         with open(code_path, 'w', encoding='utf-8') as f:
    #             f.write("# " + "="*70 + "\n")
    #             f.write(f"# Task ID: {task_id}\n")
    #             f.write(f"# Model: {model_type}\n")
    #             f.write(f"# Instruction: {instruction[:60]}...\n")
    #             f.write("# " + "="*70 + "\n#\n")
    #             f.write("# EXECUTION RESULTS:\n")
    #             f.write(f"# Syntax Valid: {'✓ YES' if syntax_valid else '✗ NO'}\n")
                
    #             if test_results:
    #                 passed = test_results.get('passed', 0)
    #                 failed = test_results.get('failed', 0)
    #                 errors = test_results.get('error', 0)
    #                 timeouts = test_results.get('timeout', 0)
    #                 pass_rate = test_results.get('pass_rate', 0.0)
                    
    #                 f.write(f"# Test Results: {'✓ PASS' if passed > 0 else '✗ FAIL'}\n")
    #                 f.write(f"#   - Passed: {passed}\n")
    #                 f.write(f"#   - Failed: {failed}\n")
    #                 f.write(f"#   - Errors: {errors}\n")
    #                 f.write(f"#   - Timeouts: {timeouts}\n")
    #                 f.write(f"#   - Pass Rate: {pass_rate:.1%}\n")
    #                 f.write(f"# Inference Time: {inference_time:.3f}s\n#\n")
                    
    #                 if 'test_results' in test_results and test_results['test_results']:
    #                     f.write("# DETAILED TEST RESULTS:\n")
    #                     for idx, result in enumerate(test_results['test_results'][:5]):
    #                         status = result['status']
    #                         status_icon = "✓" if status == "passed" else "✗"
    #                         f.write(f"# Test {idx+1}: {status_icon} {status.upper()}\n")
    #                         if result.get('message'):
    #                             error_lines = result['message'].split('\n')[:3]
    #                             for line in error_lines:
    #                                 if line.strip():
    #                                     f.write(f"#   {line[:70]}\n")
    #                     f.write("#\n")
                
    #             f.write("# " + "="*70 + "\n\n")
    #             f.write(code)
    #             f.write("\n\n")
    #             f.write("# " + "="*70 + "\n")
    #             f.write("# GEMINI-GENERATED TEST CASES\n")
    #             f.write("# " + "="*70 + "\n")
    #             f.write("# Uncomment below to run tests:\n")
    #             for line in test.split('\n'):
    #                 f.write(f"# {line}\n")
            
    #     except Exception as e:
    #         print(f"ERROR saving file for {task_id}: {e}")
    def save_generated_code_file(self, code: str, test: str, task_id: str,
                            instruction: str, model_type: str, problem_idx: int,
                            test_results: Dict = None,
                            syntax_valid: bool = True,
                            inference_time: float = 0.0,
                            execution_timing: Dict = None):  # <- ADD THIS PARAMETER
        """Save generated code to file with execution timing"""
        if not self.save_code_files:
            return
        
        try:
            safe_task_id = task_id.replace("/", "_").replace(" ", "_")
            model_dir = os.path.join(self.code_output_dir, model_type)
            
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
            
            code_filename = f"problem_{problem_idx:03d}_{safe_task_id}.py"
            code_path = os.path.join(model_dir, code_filename)
            
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write("# " + "="*70 + "\n")
                f.write(f"# Task ID: {task_id}\n")
                f.write(f"# Model: {model_type}\n")
                f.write(f"# Instruction: {instruction[:60]}...\n")
                f.write("# " + "="*70 + "\n#\n")
                f.write("# EXECUTION RESULTS:\n")
                f.write(f"# Syntax Valid: {'✓ YES' if syntax_valid else '✗ NO'}\n")
                
                # Add execution timing
                if execution_timing:
                    if execution_timing.get('success'):
                        avg_time_ms = execution_timing['avg_time'] * 1000
                        min_time_ms = execution_timing['min_time'] * 1000
                        max_time_ms = execution_timing['max_time'] * 1000
                        std_time_ms = execution_timing['std_time'] * 1000
                        
                        f.write(f"# Code Execution Time (avg): {avg_time_ms:.4f}ms\n")
                        f.write(f"#   Min: {min_time_ms:.4f}ms, Max: {max_time_ms:.4f}ms\n")
                        f.write(f"#   Std Dev: {std_time_ms:.4f}ms (over {execution_timing['num_runs']} runs)\n")
                        
                        if 'note' in execution_timing:
                            f.write(f"#   Note: {execution_timing['note']}\n")
                    else:
                        f.write(f"# Code Execution: FAILED - {execution_timing.get('error', 'Unknown error')}\n")
                
                if test_results:
                    passed = test_results.get('passed', 0)
                    failed = test_results.get('failed', 0)
                    errors = test_results.get('error', 0)
                    timeouts = test_results.get('timeout', 0)
                    pass_rate = test_results.get('pass_rate', 0.0)
                    test_exec_time = test_results.get('execution_time', 0.0)
                    
                    f.write(f"#\n# Test Results: {'✓ PASS' if passed > 0 else '✗ FAIL'}\n")
                    f.write(f"#   - Passed: {passed}\n")
                    f.write(f"#   - Failed: {failed}\n")
                    f.write(f"#   - Errors: {errors}\n")
                    f.write(f"#   - Timeouts: {timeouts}\n")
                    f.write(f"#   - Pass Rate: {pass_rate:.1%}\n")
                    f.write(f"#   - Test Execution Time: {test_exec_time:.3f}s\n")
                    f.write(f"#\n# Model Inference Time: {inference_time:.3f}s\n#\n")
                    
                    if 'test_results' in test_results and test_results['test_results']:
                        f.write("# DETAILED TEST RESULTS:\n")
                        for idx, result in enumerate(test_results['test_results'][:5]):
                            status = result['status']
                            status_icon = "✓" if status == "passed" else "✗"
                            f.write(f"# Test {idx+1}: {status_icon} {status.upper()}\n")
                            if result.get('message'):
                                error_lines = result['message'].split('\n')[:3]
                                for line in error_lines:
                                    if line.strip():
                                        f.write(f"#   {line[:70]}\n")
                        f.write("#\n")
                else:
                    f.write("# Test Results: Not executed\n")
                    f.write(f"# Model Inference Time: {inference_time:.3f}s\n#\n")
                
                f.write("# " + "="*70 + "\n\n")
                f.write(code)
                f.write("\n\n")
                f.write("# " + "="*70 + "\n")
                f.write("# GEMINI-GENERATED TEST CASES\n")
                f.write("# " + "="*70 + "\n")
                f.write("# Uncomment below to run tests:\n")
                for line in test.split('\n'):
                    f.write(f"# {line}\n")
            
            return code_path
            
        except Exception as e:
            print(f"ERROR saving file for {task_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    def calculate_final_metrics(self):
        """Calculate and display comprehensive metrics for all 6 versions"""
        print("\n" + "="*70)
        print("INSTRUCTCODER BENCHMARK RESULTS - 6 VARIANTS")
        print("="*70)
        
        if not self.results.get('baseline_simple'):
            print("No results to display")
            return
        
        total = len(self.results['baseline_simple'])
        
        # Syntax validity
        baseline_simple_syntax = np.mean([r['syntax_valid'] for r in self.results['baseline_simple']])
        baseline_clean_syntax = np.mean([r['syntax_valid'] for r in self.results['baseline_clean']])
        graph_with_simple_syntax = np.mean([r['syntax_valid'] for r in self.results['graph_with_graphs_simple']])
        graph_with_clean_syntax = np.mean([r['syntax_valid'] for r in self.results['graph_with_graphs_clean']])
        graph_no_simple_syntax = np.mean([r['syntax_valid'] for r in self.results['graph_no_graphs_simple']])
        graph_no_clean_syntax = np.mean([r['syntax_valid'] for r in self.results['graph_no_graphs_clean']])
        
        print(f"\n📋 SYNTAX VALIDITY:")
        print(f"  Baseline Simple:              {baseline_simple_syntax:.3f}")
        print(f"  Baseline Clean:               {baseline_clean_syntax:.3f}")
        print(f"  Graph(WITH graphs) Simple:    {graph_with_simple_syntax:.3f}")
        print(f"  Graph(WITH graphs) Clean:     {graph_with_clean_syntax:.3f}")
        print(f"  Graph(NO graphs) Simple:      {graph_no_simple_syntax:.3f}")
        print(f"  Graph(NO graphs) Clean:       {graph_no_clean_syntax:.3f}")
        
        # Pass@1
        baseline_simple_pass1 = np.mean([r['pass_rate'] for r in self.results['baseline_simple']])
        baseline_clean_pass1 = np.mean([r['pass_rate'] for r in self.results['baseline_clean']])
        graph_with_simple_pass1 = np.mean([r['pass_rate'] for r in self.results['graph_with_graphs_simple']])
        graph_with_clean_pass1 = np.mean([r['pass_rate'] for r in self.results['graph_with_graphs_clean']])
        graph_no_simple_pass1 = np.mean([r['pass_rate'] for r in self.results['graph_no_graphs_simple']])
        graph_no_clean_pass1 = np.mean([r['pass_rate'] for r in self.results['graph_no_graphs_clean']])
        
        print(f"\n🎯 PASS@1:")
        print(f"  Baseline Simple:              {baseline_simple_pass1:.3f}")
        print(f"  Baseline Clean:               {baseline_clean_pass1:.3f}")
        print(f"  Graph(WITH graphs) Simple:    {graph_with_simple_pass1:.3f}")
        print(f"  Graph(WITH graphs) Clean:     {graph_with_clean_pass1:.3f}")
        print(f"  Graph(NO graphs) Simple:      {graph_no_simple_pass1:.3f}")
        print(f"  Graph(NO graphs) Clean:       {graph_no_clean_pass1:.3f}")
        
        print(f"\n📈 KEY IMPROVEMENTS (using Clean versions):")
        print(f"  Graph(WITH) vs Baseline:       {(graph_with_clean_pass1-baseline_clean_pass1):.3f} ({((graph_with_clean_pass1-baseline_clean_pass1)/max(baseline_clean_pass1,0.001)*100):+.1f}%)")
        print(f"  Graph(NO) vs Baseline:         {(graph_no_clean_pass1-baseline_clean_pass1):.3f} ({((graph_no_clean_pass1-baseline_clean_pass1)/max(baseline_clean_pass1,0.001)*100):+.1f}%)")
        print(f"  Graph(WITH) vs Graph(NO):      {(graph_with_clean_pass1-graph_no_clean_pass1):.3f} ({((graph_with_clean_pass1-graph_no_clean_pass1)/max(graph_no_clean_pass1,0.001)*100):+.1f}%) ⭐ GRAPH CONTRIBUTION")
        
        print(f"\n📈 EXTRACTION METHOD IMPACT:")
        print(f"  Baseline (Clean - Simple):     {(baseline_clean_pass1-baseline_simple_pass1):.3f}")
        print(f"  Graph-WITH (Clean - Simple):   {(graph_with_clean_pass1-graph_with_simple_pass1):.3f}")
        print(f"  Graph-NO (Clean - Simple):     {(graph_no_clean_pass1-graph_no_simple_pass1):.3f}")
        
        # Perfect solutions
        baseline_simple_perfect = sum(1 for r in self.results['baseline_simple'] if r['pass_rate'] == 1.0)
        baseline_clean_perfect = sum(1 for r in self.results['baseline_clean'] if r['pass_rate'] == 1.0)
        graph_with_simple_perfect = sum(1 for r in self.results['graph_with_graphs_simple'] if r['pass_rate'] == 1.0)
        graph_with_clean_perfect = sum(1 for r in self.results['graph_with_graphs_clean'] if r['pass_rate'] == 1.0)
        graph_no_simple_perfect = sum(1 for r in self.results['graph_no_graphs_simple'] if r['pass_rate'] == 1.0)
        graph_no_clean_perfect = sum(1 for r in self.results['graph_no_graphs_clean'] if r['pass_rate'] == 1.0)
        
        print(f"\n✨ PERFECT SOLUTIONS:")
        print(f"  Baseline Simple:              {baseline_simple_perfect}/{total} ({baseline_simple_perfect/total*100:.1f}%)")
        print(f"  Baseline Clean:               {baseline_clean_perfect}/{total} ({baseline_clean_perfect/total*100:.1f}%)")
        print(f"  Graph(WITH graphs) Simple:    {graph_with_simple_perfect}/{total} ({graph_with_simple_perfect/total*100:.1f}%)")
        print(f"  Graph(WITH graphs) Clean:     {graph_with_clean_perfect}/{total} ({graph_with_clean_perfect/total*100:.1f}%)")
        print(f"  Graph(NO graphs) Simple:      {graph_no_simple_perfect}/{total} ({graph_no_simple_perfect/total*100:.1f}%)")
        print(f"  Graph(NO graphs) Clean:       {graph_no_clean_perfect}/{total} ({graph_no_clean_perfect/total*100:.1f}%)")
        
        # Test code generation
        baseline_had_tests = sum(1 for r in self.results['baseline_clean'] if r.get('had_test_code', False))
        graph_with_had_tests = sum(1 for r in self.results['graph_with_graphs_clean'] if r.get('had_test_code', False))
        graph_no_had_tests = sum(1 for r in self.results['graph_no_graphs_clean'] if r.get('had_test_code', False))
        
        print(f"\n🧪 TEST CODE GENERATION (detected in clean version):")
        print(f"  Baseline:              {baseline_had_tests}/{total} ({baseline_had_tests/total*100:.1f}%)")
        print(f"  Graph(WITH graphs):    {graph_with_had_tests}/{total} ({graph_with_had_tests/total*100:.1f}%)")
        print(f"  Graph(NO graphs):      {graph_no_had_tests}/{total} ({graph_no_had_tests/total*100:.1f}%)")
        
        # Inference time
        baseline_time = np.mean([r['inference_time'] for r in self.results['baseline_simple']])
        graph_with_time = np.mean([r['inference_time'] for r in self.results['graph_with_graphs_simple']])
        graph_no_time = np.mean([r['inference_time'] for r in self.results['graph_no_graphs_simple']])
        
        print(f"\n⏱️  AVERAGE INFERENCE TIME:")
        print(f"  Baseline:              {baseline_time:.3f}s")
        print(f"  Graph(WITH graphs):    {graph_with_time:.3f}s")
        print(f"  Graph(NO graphs):      {graph_no_time:.3f}s")
        
        # Test statistics
        avg_tests = np.mean([r['num_tests'] for r in self.results['baseline_simple']])
        print(f"\n🧪 AVERAGE GEMINI-GENERATED TESTS PER PROBLEM: {avg_tests:.1f}")
        
        print(f"\n{'='*70}")
    
    def save_results(self):
        """Save comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"instructcoder_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results as JSON
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create detailed summary
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            f.write("INSTRUCTCODER BENCHMARK SUMMARY - 6 VARIANTS\n")
            f.write("="*70 + "\n\n")
            
            total = len(self.results.get('baseline_simple', []))
            f.write(f"Total problems evaluated: {total}\n\n")
            
            if total > 0:
                # Pass@1 scores for clean versions (most important)
                baseline_clean_pass1 = np.mean([r['pass_rate'] for r in self.results['baseline_clean']])
                graph_with_clean_pass1 = np.mean([r['pass_rate'] for r in self.results['graph_with_graphs_clean']])
                graph_no_clean_pass1 = np.mean([r['pass_rate'] for r in self.results['graph_no_graphs_clean']])
                
                f.write("PASS@1 SCORES (Clean Extraction):\n")
                f.write(f"  Baseline:              {baseline_clean_pass1:.3f}\n")
                f.write(f"  Graph (WITH graphs):   {graph_with_clean_pass1:.3f}\n")
                f.write(f"  Graph (NO graphs):     {graph_no_clean_pass1:.3f}\n\n")
                
                f.write("KEY FINDINGS:\n")
                f.write(f"  Graph vs Baseline improvement:  {(graph_with_clean_pass1 - baseline_clean_pass1):.3f}\n")
                f.write(f"  Graph contribution (WITH - NO): {(graph_with_clean_pass1 - graph_no_clean_pass1):.3f}\n\n")
                
                # Perfect solutions
                baseline_clean_perfect = sum(1 for r in self.results['baseline_clean'] if r['pass_rate'] == 1.0)
                graph_with_clean_perfect = sum(1 for r in self.results['graph_with_graphs_clean'] if r['pass_rate'] == 1.0)
                graph_no_clean_perfect = sum(1 for r in self.results['graph_no_graphs_clean'] if r['pass_rate'] == 1.0)
                
                f.write("PERFECT SOLUTIONS (Clean versions):\n")
                f.write(f"  Baseline:              {baseline_clean_perfect}/{total} ({baseline_clean_perfect/total*100:.1f}%)\n")
                f.write(f"  Graph (WITH graphs):   {graph_with_clean_perfect}/{total} ({graph_with_clean_perfect/total*100:.1f}%)\n")
                f.write(f"  Graph (NO graphs):     {graph_no_clean_perfect}/{total} ({graph_no_clean_perfect/total*100:.1f}%)\n\n")
                
                # Test info
                avg_tests = np.mean([r['num_tests'] for r in self.results['baseline_simple']])
                f.write(f"Average Gemini-generated tests per problem: {avg_tests:.1f}\n")
        
        print(f"\n✅ Results saved to {output_dir}/")
        if self.save_code_files:
            print(f"✅ Generated code (6 variants) saved to {self.code_output_dir}/")

def run_instructcoder_with_gemini():
    """Main function to run InstructCoder benchmark with Gemini"""
    
    # # Get Gemini API key from environment or input
    # gemini_api_key = os.getenv("GEMINI_API_KEY")
    # if not gemini_api_key:
    #     gemini_api_key = input("Enter your Gemini API key: ")
    
    benchmark = InstructCoderBenchmarkWithGemini(
        baseline_model_path="/home/xuhaoche/.llama/HF/Llama3.1-8B-Instruct",
        graph_model_checkpoint_path="/home/xuhaoche/GACO/checkpoints_graph_lora/0910_layer1/final",
        processed_data_dir="processed_data/training_data",
        gemini_api_key="AIzaSyDtr2GqqhJXSkp_SJ5StJN6JGp3tA8QEHo",
        target_layers=[0],
        device='cuda',
        num_samples=50,  # Start with 50 samples
        num_tests_per_problem=3,
        save_code_files=True
    )
    
    benchmark.run_benchmark()


if __name__ == "__main__":
    run_instructcoder_with_gemini()