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
from typing import Dict, List, Tuple, Any
import difflib

# Import your existing functions
from models import LlamaWithGraphLayerSpecific
from preprocessing import ASTGraphBuilder, CFGExtractor, DFGBuilder, cfg_to_pyg_data, process_sample_correct_format
from torch_geometric.data import Data
from testing import generate_with_layerwise_graphs

def load_humaneval_local(filepath="HumanEval.jsonl.gz"):
    """Load HumanEval from local jsonl.gz file"""
    problems = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            problems.append(json.loads(line.strip()))
    return problems

import re
from typing import Tuple

class CodeExtractor:
    """Extract code from generated text - Simplified version"""
    
    @staticmethod
    def extract_python_code(text: str) -> str:
        """Extract Python code from markdown or plain text"""
        python_blocks = re.findall(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
        if python_blocks:
            return python_blocks[0].strip()
        
        generic_blocks = re.findall(r'```\s*\n(.*?)\n```', text, re.DOTALL)
        if generic_blocks:
            return generic_blocks[0].strip()
        
        return text.strip()

    @staticmethod
    def _detect_base_indent(prompt: str) -> str:
        """Detect the indentation of the code body from the prompt"""
        lines = prompt.rstrip().split('\n')
        for line in reversed(lines):
            if line.strip():  # find last non-empty line
                # get indentation of that line
                return re.match(r'^\s*', line).group(0)
        return ''
    
    @staticmethod
    def _indent_first_line_if_needed(code: str, base_indent: str) -> str:
        """Indent the first non-empty line of LLM code by one level if unindented"""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip():
                # If line has no leading spaces, indent it relative to base_indent
                if not line.startswith(' ') and not line.startswith('\t'):
                    lines[i] = base_indent + line  # add one indent level (4 spaces)
                break
        return '\n'.join(lines)

    @staticmethod
    def simple_concatenate(completion: str, prompt: str) -> str:
        """Concatenate prompt and LLM output with proper indentation"""
        code = CodeExtractor.extract_python_code(completion)
        if not code.strip():
            return prompt
        
        base_indent = CodeExtractor._detect_base_indent(prompt)
        code = CodeExtractor._indent_first_line_if_needed(code, base_indent)
        
        full_code = prompt.rstrip() + '\n' + code
        return full_code

    @staticmethod
    def extract_until_test_markers(completion: str, prompt: str) -> Tuple[str, bool]:
        """Stop at test markers but keep LLM's original indentation"""
        code = CodeExtractor.extract_python_code(completion)
        if not code.strip():
            return prompt, False
        
        lines = code.split('\n')
        body_lines = []
        had_test_code = False
        
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('print(') or
                stripped.startswith('# Test') or
                stripped.startswith('# Example') or
                stripped.startswith('# Usage') or
                stripped.startswith('if __name__')):
                had_test_code = True
                break
            body_lines.append(line)
        
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        
        if not body_lines:
            return prompt, had_test_code
        
        base_indent = CodeExtractor._detect_base_indent(prompt)
        first_fixed = CodeExtractor._indent_first_line_if_needed('\n'.join(body_lines), base_indent)
        
        full_code = prompt.rstrip() + '\n' + first_fixed
        return full_code, had_test_code

class StandardBenchmarkEvaluator:
    """Evaluator for standard Python benchmarks"""
    
    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
        self.code_extractor = CodeExtractor()
    def measure_code_execution_time(self, code: str, num_runs: int = 3) -> Dict:
        """
        Measure how long the generated code takes to execute (without tests)
        
        Args:
            code: The generated code
            num_runs: Number of times to run for average
            
        Returns:
            Dictionary with timing information
        """
        try:
            # For code that just defines functions (doesn't execute), 
            # measure the definition time only
            namespace = {}
            
            # Compile the code first
            try:
                compiled_code = compile(code, '<string>', 'exec')
            except SyntaxError as e:
                return {
                    'success': False,
                    'error': f'Syntax error: {str(e)}',
                    'avg_time': 0.0,
                    'min_time': 0.0,
                    'max_time': 0.0,
                    'std_time': 0.0,
                    'num_runs': 0
                }
            
            execution_times = []
            
            for _ in range(num_runs):
                try:
                    start = time.time()
                    exec(compiled_code, namespace)
                    execution_times.append(time.time() - start)
                except Exception as e:
                    # If execution fails, return error
                    return {
                        'success': False,
                        'error': f'Execution error: {str(e)}',
                        'avg_time': 0.0,
                        'min_time': 0.0,
                        'max_time': 0.0,
                        'std_time': 0.0,
                        'num_runs': 0
                    }
            
            # Check if we got any times
            if not execution_times or all(t == 0 for t in execution_times):
                # Code likely just defines functions without calling them
                return {
                    'success': True,
                    'avg_time': execution_times[0] if execution_times else 0.0,
                    'min_time': execution_times[0] if execution_times else 0.0,
                    'max_time': execution_times[0] if execution_times else 0.0,
                    'std_time': 0.0,
                    'num_runs': num_runs,
                    'note': 'Code defines functions but does not execute them'
                }
            
            return {
                'success': True,
                'avg_time': np.mean(execution_times),
                'min_time': np.min(execution_times),
                'max_time': np.max(execution_times),
                'std_time': np.std(execution_times) if len(execution_times) > 1 else 0.0,
                'num_runs': num_runs
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'avg_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'std_time': 0.0,
                'num_runs': 0
            }
            
        
    # def evaluate_functional_correctness(self, 
    #                                    generated_code: str,
    #                                    test_cases: List[Dict],
    #                                    timeout: int = 5) -> Dict:
    #     """
    #     Evaluate functional correctness using test cases
    #     Returns pass rate and detailed results
    #     """
    #     results = {
    #         'passed': 0,
    #         'failed': 0,
    #         'error': 0,
    #         'timeout': 0,
    #         'test_results': []
    #     }
        
    #     for i, test_case in enumerate(test_cases):
    #         test_result = self._run_single_test(
    #             generated_code, 
    #             test_case, 
    #             timeout
    #         )
            
    #         results['test_results'].append(test_result)
    #         results[test_result['status']] += 1
        
    #     total_tests = len(test_cases)
    #     results['pass_rate'] = results['passed'] / total_tests if total_tests > 0 else 0.0
        
    #     return results
    
    # def _run_single_test(self, code: str, test_case: Dict, timeout: int) -> Dict:
    #     """Run a single test case"""
    #     try:
    #         # Create test script
    #         # print("RUN_SINGLE_TEST_CASE")
    #         codes_dir = os.path.join(os.path.dirname(__file__), 'generated_codes')
    #         os.makedirs(codes_dir, exist_ok=True)
    #         # print(codes_dir)

    #         test_script = f"{code}\n\n{test_case['test']}"
            
    #         with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=codes_dir, encoding='utf-8') as f:
    #             f.write(test_script)
    #             temp_file = f.name
            
    #         # Run test
    #         result = subprocess.run(
    #             [sys.executable, temp_file],
    #             capture_output=True,
    #             text=True,
    #             timeout=timeout
    #         )
            
    #         os.unlink(temp_file)
            
    #         if result.returncode == 0:
    #             return {'status': 'passed', 'message': ''}
    #         else:
    #             return {'status': 'failed', 'message': result.stderr}
                
    #     except subprocess.TimeoutExpired:
    #         return {'status': 'timeout', 'message': 'Test execution timeout'}
    #     except Exception as e:
    #         return {'status': 'error', 'message': str(e)}
    
    def evaluate_functional_correctness(self, 
                                       generated_code: str,
                                       test_cases: List[Dict],
                                       timeout: int = 5) -> Dict:
        """
        Evaluate functional correctness using test cases
        Returns pass rate, detailed results, and execution time
        """
        results = {
            'passed': 0,
            'failed': 0,
            'error': 0,
            'timeout': 0,
            'test_results': [],
            'execution_time': 0.0  # Time to run the tests
        }
        
        test_start = time.time()
        
        for i, test_case in enumerate(test_cases):
            test_result = self._run_single_test(
                generated_code, 
                test_case, 
                timeout
            )
            
            results['test_results'].append(test_result)
            results[test_result['status']] += 1
        
        results['execution_time'] = time.time() - test_start
        
        total_tests = len(test_cases)
        results['pass_rate'] = results['passed'] / total_tests if total_tests > 0 else 0.0
        
        return results
    
    def _run_single_test(self, code: str, test_case: Dict, timeout: int) -> Dict:
        """Run a single test case"""
        try:
            codes_dir = os.path.join(os.path.dirname(__file__), 'generated_codes')
            os.makedirs(codes_dir, exist_ok=True)

            test_script = f"{code}\n\n{test_case['test']}"
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=codes_dir, encoding='utf-8') as f:
                f.write(test_script)
                temp_file = f.name
            
            # Run test
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return {'status': 'passed', 'message': ''}
            else:
                return {'status': 'failed', 'message': result.stderr}
                
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'message': 'Test execution timeout'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    def calculate_pass_at_k(self, n: int, c: int, k: int) -> float:
        """
        Calculate pass@k metric
        n: total samples
        c: number of correct samples  
        k: k value
        """
        if n - c < k:
            return 1.0
        return 1.0 - (np.prod([1.0 - k / (n - i) for i in range(c)]))
    
    def syntax_check(self, code: str) -> bool:
        """Check if code is syntactically valid"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False


class HumanEvalBenchmark:
    """HumanEval benchmark handler"""
    def __init__(self, 
             baseline_model_path: str,
             graph_model_checkpoint_path: str,
             processed_data_dir: str,
             target_layers: List[int] = [0],
             device: str = 'cuda',
             num_samples: int = None,
             save_code_files: bool = True):
    
        self.baseline_model_path = baseline_model_path
        self.graph_model_checkpoint_path = graph_model_checkpoint_path
        self.processed_data_dir = processed_data_dir
        self.target_layers = target_layers
        self.device = device
        self.num_samples = num_samples
        self.save_code_files = save_code_files
        
        self.evaluator = StandardBenchmarkEvaluator('humaneval')
        self.code_extractor = CodeExtractor()
        
        # Results storage
        self.results = {
            'baseline_simple': [],
            'baseline_clean': [],
            'graph_simple': [],
            'graph_clean': [],
            'metadata': []
        }
        
        # Create output directory for generated code IMMEDIATELY
        if self.save_code_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.code_output_dir = f"generated_code_humaneval_{timestamp}"
            
            # Create ALL directories upfront
            dirs_to_create = [
                "baseline_simple",
                "baseline_clean", 
                "graph_simple",
                "graph_clean",
                "tests"
            ]
            
            for dir_name in dirs_to_create:
                dir_path = os.path.join(self.code_output_dir, dir_name)
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
            
            print(f"\n✓ All output directories created in: {self.code_output_dir}")

            
        def calculate_code_similarity(self, code1: str, code2: str) -> float:
            """Calculate similarity between two code snippets"""
            from difflib import SequenceMatcher
            return SequenceMatcher(None, code1, code2).ratio()
        
        def analyze_code_differences(self, baseline_code: str, graph_code: str) -> Dict:
            """Analyze differences between baseline and graph code"""
            import difflib
            
            baseline_lines = baseline_code.splitlines()
            graph_lines = graph_code.splitlines()
            
            # Get unified diff
            diff = list(difflib.unified_diff(
                baseline_lines, graph_lines,
                lineterm='', 
                fromfile='baseline', 
                tofile='graph'
            ))
            
            # Count changes
            additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
            deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
            
            return {
                'num_additions': additions,
                'num_deletions': deletions,
                'total_changes': additions + deletions,
                'diff_lines': diff[:20]  # First 20 lines of diff
            }
    
    def create_comparison_report(self):
        """Create detailed comparison report"""
        output_dir = self.code_output_dir if self.save_code_files else "humaneval_results"
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, "detailed_comparison.md")
        
        with open(report_path, 'w') as f:
            f.write("# HumanEval Detailed Comparison: Baseline vs Graph Model\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall statistics
            f.write("## Overall Statistics\n\n")
            
            total = len(self.results['baseline'])
            baseline_passed = sum(1 for r in self.results['baseline'] if r['passed'] > 0)
            graph_passed = sum(1 for r in self.results['graph_model'] if r['passed'] > 0)
            
            baseline_syntax = sum(1 for r in self.results['baseline'] if r['syntax_valid'])
            graph_syntax = sum(1 for r in self.results['graph_model'] if r['syntax_valid'])
            
            f.write(f"| Metric | Baseline | Graph | Difference |\n")
            f.write(f"|--------|----------|-------|------------|\n")
            f.write(f"| Total Problems | {total} | {total} | - |\n")
            f.write(f"| Syntax Valid | {baseline_syntax} ({baseline_syntax/total*100:.1f}%) | {graph_syntax} ({graph_syntax/total*100:.1f}%) | {graph_syntax-baseline_syntax:+d} |\n")
            f.write(f"| Tests Passed | {baseline_passed} ({baseline_passed/total*100:.1f}%) | {graph_passed} ({graph_passed/total*100:.1f}%) | {graph_passed-baseline_passed:+d} |\n")
            f.write("\n")
            
            # Performance breakdown
            f.write("## Performance Breakdown\n\n")
            
            both_passed = sum(1 for m in self.results['metadata'] if m['both_passed'])
            both_failed = sum(1 for m in self.results['metadata'] if m['both_failed'])
            graph_better = sum(1 for m in self.results['metadata'] if m['graph_better'])
            baseline_better = sum(1 for m in self.results['metadata'] if m['baseline_better'])
            
            f.write(f"- **Both models passed**: {both_passed} ({both_passed/total*100:.1f}%)\n")
            f.write(f"- **Both models failed**: {both_failed} ({both_failed/total*100:.1f}%)\n")
            f.write(f"- **Graph better**: {graph_better} ({graph_better/total*100:.1f}%)\n")
            f.write(f"- **Baseline better**: {baseline_better} ({baseline_better/total*100:.1f}%)\n\n")
            
            # Code characteristics
            f.write("## Code Characteristics\n\n")
            
            baseline_avg_length = np.mean([r['code_length'] for r in self.results['baseline']])
            graph_avg_length = np.mean([r['code_length'] for r in self.results['graph_model']])
            
            baseline_avg_lines = np.mean([r['num_lines'] for r in self.results['baseline']])
            graph_avg_lines = np.mean([r['num_lines'] for r in self.results['graph_model']])
            
            avg_similarity = np.mean([m['code_similarity'] for m in self.results['metadata']])
            
            f.write(f"- Average code length: Baseline={baseline_avg_length:.0f} chars, Graph={graph_avg_length:.0f} chars\n")
            f.write(f"- Average lines: Baseline={baseline_avg_lines:.1f}, Graph={graph_avg_lines:.1f}\n")
            f.write(f"- Average code similarity: {avg_similarity:.3f}\n\n")
            
            # Cases where graph model helped
            f.write("## Cases Where Graph Model Helped\n\n")
            
            graph_wins = [
                (i, self.results['metadata'][i]) 
                for i in range(len(self.results['metadata'])) 
                if self.results['metadata'][i]['graph_better']
            ]
            
            if graph_wins:
                for i, meta in graph_wins[:10]:  # Show first 10
                    f.write(f"### Problem {i}: {meta['task_id']}\n")
                    f.write(f"- Baseline: {'PASS' if self.results['baseline'][i]['passed'] else 'FAIL'}\n")
                    f.write(f"- Graph: {'PASS' if self.results['graph_model'][i]['passed'] else 'FAIL'}\n")
                    f.write(f"- Similarity: {meta['code_similarity']:.3f}\n")
                    f.write(f"- Changes: +{meta['differences']['num_additions']} -{meta['differences']['num_deletions']} lines\n\n")
            else:
                f.write("*No cases where graph model outperformed baseline*\n\n")
            
            # Cases where baseline was better
            f.write("## Cases Where Baseline Was Better\n\n")
            
            baseline_wins = [
                (i, self.results['metadata'][i])
                for i in range(len(self.results['metadata']))
                if self.results['metadata'][i]['baseline_better']
            ]
            
            if baseline_wins:
                for i, meta in baseline_wins[:10]:  # Show first 10
                    f.write(f"### Problem {i}: {meta['task_id']}\n")
                    f.write(f"- Baseline: {'PASS' if self.results['baseline'][i]['passed'] else 'FAIL'}\n")
                    f.write(f"- Graph: {'PASS' if self.results['graph_model'][i]['passed'] else 'FAIL'}\n")
                    f.write(f"- Similarity: {meta['code_similarity']:.3f}\n")
                    f.write(f"- Changes: +{meta['differences']['num_additions']} -{meta['differences']['num_deletions']} lines\n\n")
            else:
                f.write("*No cases where baseline outperformed graph model*\n\n")
            
            # Inference time analysis
            f.write("## Inference Time Analysis\n\n")
            
            baseline_times = [r['inference_time'] for r in self.results['baseline']]
            graph_times = [r['inference_time'] for r in self.results['graph_model']]
            
            f.write(f"| Metric | Baseline | Graph |\n")
            f.write(f"|--------|----------|-------|\n")
            f.write(f"| Mean | {np.mean(baseline_times):.3f}s | {np.mean(graph_times):.3f}s |\n")
            f.write(f"| Median | {np.median(baseline_times):.3f}s | {np.median(graph_times):.3f}s |\n")
            f.write(f"| Min | {np.min(baseline_times):.3f}s | {np.min(graph_times):.3f}s |\n")
            f.write(f"| Max | {np.max(baseline_times):.3f}s | {np.max(graph_times):.3f}s |\n")
            f.write(f"| Std Dev | {np.std(baseline_times):.3f}s | {np.std(graph_times):.3f}s |\n")
        
        print(f"\n✓ Detailed comparison saved to: {report_path}")
    
    def save_generated_code_file(self, code: str, test: str, task_id: str, 
                            model_type: str, problem_idx: int,
                            test_results: Dict = None,
                            syntax_valid: bool = True,
                            inference_time: float = 0.0):
        """Save generated code to a separate Python file with test results"""
        if not self.save_code_files:
            return
        
        try:
            # Sanitize task_id for filename
            safe_task_id = task_id.replace("/", "_").replace(" ", "_")
            
            # Verify directory exists
            model_dir = os.path.join(self.code_output_dir, model_type)
            if not os.path.exists(model_dir):
                print(f"WARNING: Directory {model_dir} doesn't exist, creating it...")
                os.makedirs(model_dir, exist_ok=True)
            
            # Save the generated code
            code_filename = f"problem_{problem_idx:03d}_{safe_task_id}.py"
            code_path = os.path.join(model_dir, code_filename)
            
            # print(f"Saving to: {code_path}")
            
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write("# " + "="*70 + "\n")
                f.write(f"# Task ID: {task_id}\n")
                f.write(f"# Model: {model_type}\n")
                f.write(f"# Problem: {problem_idx}\n")
                f.write("# " + "="*70 + "\n")
                f.write("#\n")
                f.write("# EXECUTION RESULTS:\n")
                f.write(f"# Syntax Valid: {'✓ YES' if syntax_valid else '✗ NO'}\n")
                
                if test_results:
                    passed = test_results.get('passed', 0)
                    failed = test_results.get('failed', 0)
                    errors = test_results.get('error', 0)
                    timeouts = test_results.get('timeout', 0)
                    pass_rate = test_results.get('pass_rate', 0.0)
                    
                    f.write(f"# Test Results: {'✓ PASS' if passed > 0 else '✗ FAIL'}\n")
                    f.write(f"#   - Passed: {passed}\n")
                    f.write(f"#   - Failed: {failed}\n")
                    f.write(f"#   - Errors: {errors}\n")
                    f.write(f"#   - Timeouts: {timeouts}\n")
                    f.write(f"#   - Pass Rate: {pass_rate:.1%}\n")
                    f.write(f"# Inference Time: {inference_time:.3f}s\n")
                    f.write("#\n")
                    
                    # Add detailed test results
                    if 'test_results' in test_results:
                        f.write("# DETAILED TEST RESULTS:\n")
                        for i, result in enumerate(test_results['test_results']):
                            status = result['status']
                            status_icon = "✓" if status == "passed" else "✗"
                            f.write(f"# Test {i+1}: {status_icon} {status.upper()}\n")
                            
                            if result.get('message'):
                                # Format error message nicely
                                error_lines = result['message'].split('\n')
                                for line in error_lines[:5]:  # Show first 5 lines of error
                                    if line.strip():
                                        f.write(f"#   {line}\n")
                        f.write("#\n")
                else:
                    f.write("# Test Results: Not executed\n")
                    f.write("#\n")
                
                f.write("# " + "="*70 + "\n\n")
                
                # Write the actual code
                f.write(code)
                f.write("\n\n")
                
                f.write("# " + "="*70 + "\n")
                f.write("# TEST CASES\n")
                f.write("# " + "="*70 + "\n")
                f.write("# Uncomment below to run tests:\n")
                for line in test.split('\n'):
                    f.write(f"# {line}\n")
            
            # Also save a standalone test file (runnable)
            test_filename = f"problem_{problem_idx:03d}_{safe_task_id}_test.py"
            test_path = os.path.join(self.code_output_dir, "tests", test_filename)
            
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write("# " + "="*70 + "\n")
                f.write(f"# RUNNABLE TEST FILE - {task_id}\n")
                f.write("# " + "="*70 + "\n")
                f.write(f"# Model: {model_type}\n")
                
                if test_results:
                    f.write(f"# Previous Result: {'PASS ✓' if test_results.get('passed', 0) > 0 else 'FAIL ✗'}\n")
                
                f.write("# " + "="*70 + "\n\n")
                f.write(code)
                f.write("\n\n")
                f.write("# " + "="*70 + "\n")
                f.write("# TESTS\n")
                f.write("# " + "="*70 + "\n\n")
                f.write(test)
            
            return code_path
            
        except Exception as e:
            print(f"ERROR saving file for {task_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

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
            print("Graph components loaded")
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
    
    def create_prompt(self, task: Dict) -> str:
        """Create prompt for HumanEval task"""
        prompt = task['prompt']
        
        # Add instruction for the model
        instruction = "Complete the following Python function:\n\n"
        return instruction + prompt
    
    def generate_baseline(self, model, tokenizer, prompt: str, 
                         max_new_tokens: int = 384) -> str:
        """Generate completion using baseline model"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
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
    
    def build_graphs_from_prompt(self, prompt: str, feature_mappings: Dict):
        """Build graphs from the function signature in the prompt"""
        try:
            # Extract AST from prompt (function signature)
            ast_builder = ASTGraphBuilder()
            ast_nodes, ast_edges = ast_builder.build(prompt)
            
            if len(ast_nodes) > 0:
                ast_type2id = feature_mappings['ast_type2id']
                ast_num_classes = feature_mappings['ast_num_classes']
                
                # Create node features
                indices = [ast_type2id.get(typ, 0) for typ in ast_nodes]
                x_ast = torch.nn.functional.one_hot(
                    torch.tensor(indices), 
                    num_classes=ast_num_classes
                ).float()
                
                # Pad to target dimension
                target_dim = feature_mappings['target_dim']
                if x_ast.shape[1] < target_dim:
                    pad = torch.zeros(x_ast.shape[0], target_dim - x_ast.shape[1])
                    x_ast = torch.cat([x_ast, pad], dim=1)
                elif x_ast.shape[1] > target_dim:
                    x_ast = x_ast[:, :target_dim]
                
                edge_index = torch.tensor(ast_edges, dtype=torch.long).t().contiguous() if ast_edges else torch.empty((2, 0), dtype=torch.long)
                ast_batch = Data(x=x_ast, edge_index=edge_index).to(self.device)
            else:
                ast_batch = None
            
            # Build CFG from prompt
            try:
                cfg_batch = cfg_to_pyg_data(prompt).to(self.device)
            except:
                cfg_batch = None
            
            # Build DFG from prompt
            dfg_builder = DFGBuilder()
            dfg_nodes, dfg_edges = dfg_builder.build(prompt)
            
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
            print(f"Warning: Could not build graphs from prompt: {e}")
            return None, None, None
    
    def generate_graph(self, model, tokenizer, prompt: str, 
                      feature_mappings: Dict,
                      max_new_tokens: int = 384) -> str:
        """Generate completion using graph model with graph integration"""
        
        # Build graphs from the function signature in prompt
        ast_batch, cfg_batch, dfg_batch = self.build_graphs_from_prompt(prompt, feature_mappings)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            # Use graphs during generation
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
    def run_benchmark(self):
        """Run HumanEval benchmark with simple concatenation"""
        print("Loading HumanEval dataset...")
        
        dataset = load_humaneval_local("HumanEval.jsonl.gz")
        print(f"Evaluating on {len(dataset)} problems")
        
        # Load models
        (baseline_model, baseline_tokenizer,
        graph_model, graph_tokenizer,
        feature_mappings) = self.load_models()
        
        start_time = time.time()
        
        for i, task in enumerate(tqdm(dataset, desc="Evaluating HumanEval")):
            try:
                task_id = task['task_id']
                prompt = task['prompt']
                full_prompt = self.create_prompt(task)
                canonical_solution = task['canonical_solution']
                test_cases = task['test']
                entry_point = task['entry_point']
                
                print(f"\n=== Sample {i} ===")
                # print(f"Prompt:\n{prompt}")
                
                # Generate completions
                baseline_start = time.time()
                baseline_completion = self.generate_baseline(
                    baseline_model, baseline_tokenizer, full_prompt
                )
                baseline_time = time.time() - baseline_start
                
                graph_start = time.time()
                graph_completion = self.generate_graph(
                    graph_model, graph_tokenizer, full_prompt, feature_mappings
                )
                graph_time = time.time() - graph_start
                
                # SIMPLE EXTRACTION: Just concatenate prompt + completion
                baseline_code_simple = self.code_extractor.simple_concatenate(
                    baseline_completion, prompt
                )
                
                graph_code_simple = self.code_extractor.simple_concatenate(
                    graph_completion, prompt
                )
                
                # ALTERNATIVE: Stop at test markers (optional)
                baseline_code_clean, baseline_had_tests = self.code_extractor.extract_until_test_markers(
                    baseline_completion, prompt
                )
                
                graph_code_clean, graph_had_tests = self.code_extractor.extract_until_test_markers(
                    graph_completion, prompt
                )
                
                # print(f"\nBaseline completion:\n{baseline_completion}...")
                # print(f"\nBaseline SIMPLE:\n{baseline_code_simple}...")
                # print(f"\nBaseline CLEAN (stopped at tests): {baseline_had_tests}\n{baseline_code_clean}...")
                
                # print(f"\nGraph completion:\n{graph_completion}...")
                # print(f"\nGraph SIMPLE:\n{graph_code_simple}...")
                # print(f"\nGraph CLEAN (stopped at tests): {graph_had_tests}\n{graph_code_clean}...")
                # print("="*50)
                
                # Check syntax
                baseline_simple_syntax = self.evaluator.syntax_check(baseline_code_simple)
                baseline_clean_syntax = self.evaluator.syntax_check(baseline_code_clean)
                graph_simple_syntax = self.evaluator.syntax_check(graph_code_simple)
                graph_clean_syntax = self.evaluator.syntax_check(graph_code_clean)
                
                # Prepare test cases
                test_list = [{'test': test_cases}]
                
                # Evaluate all versions
                baseline_simple_results = self.evaluator.evaluate_functional_correctness(
                    baseline_code_simple, test_list
                )
                baseline_clean_results = self.evaluator.evaluate_functional_correctness(
                    baseline_code_clean, test_list
                )
                graph_simple_results = self.evaluator.evaluate_functional_correctness(
                    graph_code_simple, test_list
                )
                graph_clean_results = self.evaluator.evaluate_functional_correctness(
                    graph_code_clean, test_list
                )


                
                print(f"\nBaseline SIMPLE - Syntax: {baseline_simple_syntax}, Pass: {baseline_simple_results['passed']}")
                print(f"Baseline CLEAN - Syntax: {baseline_clean_syntax}, Pass: {baseline_clean_results['passed']}")
                print(f"Graph SIMPLE - Syntax: {graph_simple_syntax}, Pass: {graph_simple_results['passed']}")
                print(f"Graph CLEAN - Syntax: {graph_clean_syntax}, Pass: {graph_clean_results['passed']}")
                print(f"\nBaseline SIMPLE - Result: {baseline_simple_results}")
                print(f"Baseline CLEAN - Result: {baseline_clean_results}")
                print(f"Graph SIMPLE - Result: {graph_simple_results}")
                print(f"Graph CLEAN - Result: {graph_clean_results}")
                
                # # Calculate similarities
                # similarity = self.calculate_code_similarity(baseline_code_clean, graph_code_clean)
                # differences = self.analyze_code_differences(baseline_code_clean, graph_code_clean)
                
                # Save code files
                if self.save_code_files:
                    # Save baseline versions
                    self.save_generated_code_file(
                        code=baseline_code_simple,
                        test=test_cases,
                        task_id=task_id,
                        model_type="baseline_simple",
                        problem_idx=i,
                        test_results=baseline_simple_results,
                        syntax_valid=baseline_simple_syntax,
                        inference_time=baseline_time
                    )
                    
                    self.save_generated_code_file(
                        code=baseline_code_clean,
                        test=test_cases,
                        task_id=task_id,
                        model_type="baseline_clean",
                        problem_idx=i,
                        test_results=baseline_clean_results,
                        syntax_valid=baseline_clean_syntax,
                        inference_time=baseline_time
                    )
                    
                    # Save graph versions
                    self.save_generated_code_file(
                        code=graph_code_simple,
                        test=test_cases,
                        task_id=task_id,
                        model_type="graph_simple",
                        problem_idx=i,
                        test_results=graph_simple_results,
                        syntax_valid=graph_simple_syntax,
                        inference_time=graph_time
                    )
                    
                    self.save_generated_code_file(
                        code=graph_code_clean,
                        test=test_cases,
                        task_id=task_id,
                        model_type="graph_clean",
                        problem_idx=i,
                        test_results=graph_clean_results,
                        syntax_valid=graph_clean_syntax,
                        inference_time=graph_time
                    )
                
                # Store results
                self.results['baseline_simple'] = self.results.get('baseline_simple', [])
                self.results['baseline_clean'] = self.results.get('baseline_clean', [])
                self.results['graph_simple'] = self.results.get('graph_simple', [])
                self.results['graph_clean'] = self.results.get('graph_clean', [])
                
                self.results['baseline_simple'].append({
                    'task_id': task_id,
                    'syntax_valid': baseline_simple_syntax,
                    'pass_rate': baseline_simple_results['pass_rate'],
                    'passed': baseline_simple_results.get('passed', 0),
                    'failed': baseline_simple_results.get('failed', 0),
                    'inference_time': baseline_time,
                    'extracted_code': baseline_code_simple,
                    'code_length': len(baseline_code_simple),
                    'num_lines': len(baseline_code_simple.splitlines()),
                })
                
                self.results['baseline_clean'].append({
                    'task_id': task_id,
                    'syntax_valid': baseline_clean_syntax,
                    'pass_rate': baseline_clean_results['pass_rate'],
                    'passed': baseline_clean_results.get('passed', 0),
                    'failed': baseline_clean_results.get('failed', 0),
                    'inference_time': baseline_time,
                    'extracted_code': baseline_code_clean,
                    'code_length': len(baseline_code_clean),
                    'num_lines': len(baseline_code_clean.splitlines()),
                    'had_test_code': baseline_had_tests
                })
                
                self.results['graph_simple'].append({
                    'task_id': task_id,
                    'syntax_valid': graph_simple_syntax,
                    'pass_rate': graph_simple_results['pass_rate'],
                    'passed': graph_simple_results.get('passed', 0),
                    'failed': graph_simple_results.get('failed', 0),
                    'inference_time': graph_time,
                    'extracted_code': graph_code_simple,
                    'code_length': len(graph_code_simple),
                    'num_lines': len(graph_code_simple.splitlines()),
                })
                
                self.results['graph_clean'].append({
                    'task_id': task_id,
                    'syntax_valid': graph_clean_syntax,
                    'pass_rate': graph_clean_results['pass_rate'],
                    'passed': graph_clean_results.get('passed', 0),
                    'failed': graph_clean_results.get('failed', 0),
                    'inference_time': graph_time,
                    'extracted_code': graph_code_clean,
                    'code_length': len(graph_code_clean),
                    'num_lines': len(graph_code_clean.splitlines()),
                    'had_test_code': graph_had_tests
                })
                
                self.results['metadata'].append({
                    'task_id': task_id,
                    'prompt': prompt,
                    'canonical_solution': canonical_solution,
                    # 'code_similarity': similarity,
                    # 'differences': differences,
                    'baseline_had_tests': baseline_had_tests,
                    'graph_had_tests': graph_had_tests
                })
                
            except Exception as e:
                print(f"Error on task {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        total_time = time.time() - start_time
        print(f"\nBenchmark completed in {total_time:.2f} seconds")
        
        self.calculate_final_metrics()
        self.save_results()
        

    
    def print_interim_results(self, num_processed: int):
        """Print interim results"""
        if not self.results['baseline']:
            return
        
        baseline_syntax = np.mean([r['syntax_valid'] for r in self.results['baseline']])
        graph_syntax = np.mean([r['syntax_valid'] for r in self.results['graph_model']])
        
        baseline_pass = np.mean([r['pass_rate'] for r in self.results['baseline']])
        graph_pass = np.mean([r['pass_rate'] for r in self.results['graph_model']])
        
        print(f"\n--- Interim Results (after {num_processed} problems) ---")
        print(f"Syntax Valid - Baseline: {baseline_syntax:.3f}, Graph: {graph_syntax:.3f}")
        print(f"Pass Rate    - Baseline: {baseline_pass:.3f}, Graph: {graph_pass:.3f}")
   
    def calculate_final_metrics(self):
        """Calculate and display comprehensive final metrics"""
        print("\n" + "="*70)
        print("HUMANEVAL BENCHMARK RESULTS - COMPARISON")
        print("="*70)
        
        # Check which versions we have
        if not self.results.get('baseline_simple'):
            print("No results to display")
            return
        
        total = len(self.results['baseline_simple'])
        
        # Syntax validity
        baseline_simple_syntax = np.mean([r['syntax_valid'] for r in self.results['baseline_simple']])
        baseline_clean_syntax = np.mean([r['syntax_valid'] for r in self.results['baseline_clean']])
        graph_simple_syntax = np.mean([r['syntax_valid'] for r in self.results['graph_simple']])
        graph_clean_syntax = np.mean([r['syntax_valid'] for r in self.results['graph_clean']])
        
        print(f"\nSYNTAX VALIDITY:")
        print(f"Baseline Simple: {baseline_simple_syntax:.3f}")
        print(f"Baseline Clean:  {baseline_clean_syntax:.3f}")
        print(f"Graph Simple:    {graph_simple_syntax:.3f}")
        print(f"Graph Clean:     {graph_clean_syntax:.3f}")
        
        # Pass@1
        baseline_simple_pass1 = np.mean([r['pass_rate'] for r in self.results['baseline_simple']])
        baseline_clean_pass1 = np.mean([r['pass_rate'] for r in self.results['baseline_clean']])
        graph_simple_pass1 = np.mean([r['pass_rate'] for r in self.results['graph_simple']])
        graph_clean_pass1 = np.mean([r['pass_rate'] for r in self.results['graph_clean']])
        
        print(f"\nPASS@1:")
        print(f"Baseline Simple: {baseline_simple_pass1:.3f}")
        print(f"Baseline Clean:  {baseline_clean_pass1:.3f}")
        print(f"Graph Simple:    {graph_simple_pass1:.3f}")
        print(f"Graph Clean:     {graph_clean_pass1:.3f}")
        
        print(f"\nIMPROVEMENTS:")
        print(f"Graph Simple vs Baseline Simple: {(graph_simple_pass1-baseline_simple_pass1):.3f} ({((graph_simple_pass1-baseline_simple_pass1)/max(baseline_simple_pass1,0.001)*100):+.1f}%)")
        print(f"Graph Clean vs Baseline Clean:   {(graph_clean_pass1-baseline_clean_pass1):.3f} ({((graph_clean_pass1-baseline_clean_pass1)/max(baseline_clean_pass1,0.001)*100):+.1f}%)")
        
        # Test code generation
        baseline_had_tests = sum(1 for r in self.results['baseline_clean'] if r.get('had_test_code', False))
        graph_had_tests = sum(1 for r in self.results['graph_clean'] if r.get('had_test_code', False))
        
        print(f"\nTEST CODE DETECTION (in clean version):")
        print(f"Baseline: {baseline_had_tests}/{total} ({baseline_had_tests/total*100:.1f}%)")
        print(f"Graph:    {graph_had_tests}/{total} ({graph_had_tests/total*100:.1f}%)")
        
        # Perfect solutions
        baseline_simple_perfect = sum(1 for r in self.results['baseline_simple'] if r['pass_rate'] == 1.0)
        baseline_clean_perfect = sum(1 for r in self.results['baseline_clean'] if r['pass_rate'] == 1.0)
        graph_simple_perfect = sum(1 for r in self.results['graph_simple'] if r['pass_rate'] == 1.0)
        graph_clean_perfect = sum(1 for r in self.results['graph_clean'] if r['pass_rate'] == 1.0)
        
        print(f"\nPERFECT SOLUTIONS:")
        print(f"Baseline Simple: {baseline_simple_perfect}/{total} ({baseline_simple_perfect/total*100:.1f}%)")
        print(f"Baseline Clean:  {baseline_clean_perfect}/{total} ({baseline_clean_perfect/total*100:.1f}%)")
        print(f"Graph Simple:    {graph_simple_perfect}/{total} ({graph_simple_perfect/total*100:.1f}%)")
        print(f"Graph Clean:     {graph_clean_perfect}/{total} ({graph_clean_perfect/total*100:.1f}%)")
        
        # Inference time
        baseline_avg_time = np.mean([r['inference_time'] for r in self.results['baseline_simple']])
        graph_avg_time = np.mean([r['inference_time'] for r in self.results['graph_simple']])
        
        print(f"\nAVERAGE INFERENCE TIME:")
        print(f"Baseline: {baseline_avg_time:.3f}s")
        print(f"Graph:    {graph_avg_time:.3f}s")
        print(f"Difference: {(graph_avg_time-baseline_avg_time):.3f}s ({((graph_avg_time-baseline_avg_time)/baseline_avg_time*100):+.1f}%)")

    def save_results(self):
        """Save detailed results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"humaneval_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create summary
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            f.write("HUMANEVAL BENCHMARK SUMMARY\n")
            f.write("="*50 + "\n")
            
            total = len(self.results.get('baseline_simple', []))
            f.write(f"Total problems: {total}\n\n")
            
            if total > 0:
                # Pass@1 for all versions
                baseline_simple_pass1 = np.mean([r['pass_rate'] for r in self.results['baseline_simple']])
                baseline_clean_pass1 = np.mean([r['pass_rate'] for r in self.results['baseline_clean']])
                graph_simple_pass1 = np.mean([r['pass_rate'] for r in self.results['graph_simple']])
                graph_clean_pass1 = np.mean([r['pass_rate'] for r in self.results['graph_clean']])
                
                f.write("PASS@1 SCORES:\n")
                f.write(f"Baseline Simple: {baseline_simple_pass1:.3f}\n")
                f.write(f"Baseline Clean:  {baseline_clean_pass1:.3f}\n")
                f.write(f"Graph Simple:    {graph_simple_pass1:.3f}\n")
                f.write(f"Graph Clean:     {graph_clean_pass1:.3f}\n\n")
                
                f.write("IMPROVEMENTS:\n")
                f.write(f"Graph Simple vs Baseline Simple: {(graph_simple_pass1 - baseline_simple_pass1):.3f}\n")
                f.write(f"Graph Clean vs Baseline Clean: {(graph_clean_pass1 - baseline_clean_pass1):.3f}\n\n")
                
                # Perfect solutions
                baseline_clean_perfect = sum(1 for r in self.results['baseline_clean'] if r['pass_rate'] == 1.0)
                graph_clean_perfect = sum(1 for r in self.results['graph_clean'] if r['pass_rate'] == 1.0)
                
                f.write("PERFECT SOLUTIONS (Clean versions):\n")
                f.write(f"Baseline: {baseline_clean_perfect}/{total} ({baseline_clean_perfect/total*100:.1f}%)\n")
                f.write(f"Graph: {graph_clean_perfect}/{total} ({graph_clean_perfect/total*100:.1f}%)\n")
        
        print(f"\n✓ Results saved to {output_dir}/")
        if self.save_code_files:
            print(f"✓ Generated code saved to {self.code_output_dir}/")
def run_humaneval_benchmark():
    """Main function to run HumanEval benchmark"""
    
    benchmark = HumanEvalBenchmark(
        baseline_model_path="/home/xuhaoche/.llama/HF/Llama3.1-8B-Instruct",
        graph_model_checkpoint_path="/home/xuhaoche/GACO/checkpoints_graph_lora/0910_layer1/final",
        processed_data_dir="processed_data/training_data",
        target_layers=[0],
        device='cuda',
        num_samples=164  # Full HumanEval has 164 problems, set None for all
    )
    
    benchmark.run_benchmark()

if __name__ == "__main__":
    run_humaneval_benchmark()