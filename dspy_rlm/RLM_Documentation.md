# DSPy RLM (Recursive Language Model) - Complete Documentation

## Overview

**RLM** (Recursive Language Model) is an experimental DSPy module that implements a novel inference strategy where LLMs treat long contexts as part of an **external environment** rather than feeding them directly to the model. The LLM writes Python code to programmatically examine, decompose, and recursively call sub-LLMs over snippets.

> **Reference**: "Recursive Language Models" (Zhang, Kraska, Khattab, 2025)

## How RLM Works

### Core Concept

Unlike traditional prompting where you feed the entire context to an LLM, RLM:

1. **Uses a sandboxed REPL** (Python interpreter) to let the LLM programmatically explore large contexts through code execution
2. **Iterative Execution**: The LLM writes Python code, sees the output, then decides what to do next
3. **Sub-LLM Calls**: Provides `llm_query()` and `llm_query_batched()` tools for semantic analysis of smaller chunks
4. **Final Submission**: When ready, the LLM calls `SUBMIT()` with the final output

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RLM Module                               │
├─────────────────────────────────────────────────────────────────┤
│  1. generate_action (dspy.Predict)                               │
│     - Takes: variables_info, repl_history, iteration             │
│     - Outputs: reasoning, code                                   │
├─────────────────────────────────────────────────────────────────┤
│  2. Code Interpreter (PythonInterpreter / Custom)                │
│     - Executes Python code in sandboxed environment              │
│     - Has access to: llm_query, llm_query_batched, SUBMIT, print │
├─────────────────────────────────────────────────────────────────┤
│  3. extract (dspy.Predict) - Fallback                            │
│     - Used when max_iterations reached                           │
│     - Extracts final output from trajectory                      │
└─────────────────────────────────────────────────────────────────┘
```

## Constructor Parameters

```python
class RLM(Module):
    def __init__(
        self,
        signature: type[Signature] | str,      # Defines inputs and outputs
        max_iterations: int = 20,               # Maximum REPL interaction iterations
        max_llm_calls: int = 50,                # Maximum sub-LLM calls per execution
        max_output_chars: int = 100_000,        # Max characters from REPL output
        verbose: bool = False,                  # Whether to log detailed execution
        tools: dict[str, Callable] = None,      # Additional custom tool functions
        sub_lm: dspy.LM = None,                 # LM for llm_query (can be cheaper model)
        interpreter: CodeInterpreter = None,   # Custom interpreter implementation
    ):
```

### Parameters Explained

| Parameter | Description |
|-----------|-------------|
| `signature` | String like `"context, query -> answer"` or a Signature class |
| `max_iterations` | How many times the LLM can write code and see results |
| `max_llm_calls` | Budget for `llm_query`/`llm_query_batched` calls |
| `max_output_chars` | Truncates REPL output to prevent context overflow |
| `verbose` | Set `True` to see reasoning and code at each iteration |
| `tools` | Add custom Python functions accessible in the REPL |
| `sub_lm` | Use a different (cheaper) model for sub-queries |
| `interpreter` | Provide custom CodeInterpreter (E2B, Modal, etc.) |

## Basic Usage

```python
import dspy

# Configure your LM
lm = dspy.LM('anthropic/claude-sonnet-4-5', api_key='...')
dspy.configure(lm=lm)

# Create RLM instance
rlm = dspy.RLM("context, query -> answer", max_iterations=10)

# Execute
result = rlm(
    context="...very long text...", 
    query="What is the magic number?"
)

print(result.answer)       # The final answer
print(result.trajectory)   # Full debugging trajectory
```

## Available Tools in the REPL

The LLM has access to these built-in functions:

| Tool | Description |
|------|-------------|
| `llm_query(prompt: str) -> str` | Query a sub-LLM (~500K char capacity) for semantic analysis |
| `llm_query_batched(prompts: list[str]) -> list[str]` | Query multiple prompts concurrently (much faster) |
| `print()` | **ALWAYS print to see results** - this is how the LLM observes output |
| `SUBMIT(final_outputs)` | Submit final output when done - ends execution immediately |

Plus standard Python libraries: `re`, `json`, `collections`, `math`, etc.

## The System Prompt (ACTION_INSTRUCTIONS_TEMPLATE)

Here's the actual prompt template that RLM uses:

```
You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be 
executed. You will see the output, then write more code based on what you learned. 
This is an iterative process.

Available:
- Variables: {inputs} (your input data)
- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis
- `llm_query_batched(prompts)` - query multiple prompts concurrently
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done
- Standard libraries: re, json, collections, math, etc.

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see 
the output, then you decide what to do next. Do NOT try to solve everything in one step.

1. EXPLORE FIRST - Look at your data before processing it. Print samples, check 
   types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. 
   State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), 
   reconsider your approach.
4. USE llm_query FOR SEMANTICS - String matching finds WHERE things are; 
   llm_query understands WHAT things mean.
5. MINIMIZE RETYPING - When values are long, precise, or error-prone, re-access 
   them via variables instead of retyping.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. 
   If you need to inspect printed output, run it in one step, review, then SUBMIT later.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output.
```

---

## Debugging & Inspection

### 1. Using the `trajectory` Attribute

The most important debugging tool for RLM is the **trajectory** - it contains the complete history of all REPL interactions:

```python
output = rlm(discussion=discussion, request='...')

# Access the full trajectory
for step in output.trajectory:
    print(f"=== Step ===")
    print(f"Reasoning: {step['reasoning']}")
    print(f"Code:\n{step['code']}")
    print(f"Output: {step['output']}")
    print()
```

The trajectory is a list of dictionaries, each containing:
- `reasoning`: The LLM's thinking about what to do next
- `code`: The Python code that was executed
- `output`: The result of executing that code

### 2. Using `verbose=True`

Enable verbose mode to see reasoning and code at each iteration in real-time:

```python
rlm = dspy.RLM(
    "context, query -> answer",
    verbose=True  # Logs each iteration to console
)
```

### 3. Using `dspy.inspect_history()`

Inspect the raw LLM calls (prompts and responses):

```python
# After running your RLM
output = rlm(discussion=discussion, request='...')

# See the last N LLM calls
dspy.inspect_history(n=5)
```

This shows:
- The full system message
- The messages sent to the LLM
- The raw response from the LLM

### 4. Accessing Internal Signatures

You can inspect the signatures that RLM builds:

```python
rlm = dspy.RLM("context, query -> answer")

# The action generation signature
print(rlm.generate_action.signature)

# The extract fallback signature  
print(rlm.extract.signature)
```

### 5. Custom Callbacks for Advanced Logging

Use DSPy's callback system for fine-grained logging:

```python
from dspy.utils.callback import BaseCallback

class RLMLoggingCallback(BaseCallback):
    def on_module_start(self, call_id, instance, inputs):
        print(f"Starting module: {type(instance).__name__}")
        print(f"Inputs: {inputs.keys()}")
    
    def on_module_end(self, call_id, outputs, exception):
        print(f"Outputs: {outputs.keys()}")
        if 'trajectory' in outputs:
            print(f"Trajectory length: {len(outputs['trajectory'])}")

    def on_lm_start(self, call_id, instance, inputs):
        print(f"LLM Call - Messages:\n{inputs.get('messages', [])}")
    
    def on_lm_end(self, call_id, outputs, exception):
        print(f"LLM Response: {outputs}")

# Apply the callback
dspy.configure(callbacks=[RLMLoggingCallback()])
```

### 6. Using MLflow Tracing

For production-grade observability:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("RLM_Debugging")
mlflow.dspy.autolog()

# Now all RLM calls are traced
output = rlm(context=context, query=query)
```

---

## Understanding the Output

### Prediction Object

The RLM returns a `Prediction` object with:

```python
output = rlm(discussion=discussion, request='...')

# Your output fields (defined in signature)
output.ideas          # list[str] in your example

# Debugging fields
output.trajectory     # List of all REPL interactions (list of dicts)
output.final_reasoning  # The final reasoning before SUBMIT
```

### Trajectory Structure

Each entry in `trajectory` is a dictionary with:

```python
{
    "reasoning": "The LLM's step-by-step thinking",
    "code": "print(len(discussion))",
    "output": "1234"
}
```

---

## Advanced Usage

### Adding Custom Tools

```python
def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return summary."""
    # Your implementation
    return summary

rlm = dspy.RLM(
    "question -> answer",
    tools={"search_wikipedia": search_wikipedia}
)
```

### Using a Cheaper Model for Sub-queries

```python
# Main model for reasoning
main_lm = dspy.LM('anthropic/claude-sonnet-4-5')

# Cheaper model for llm_query calls
sub_lm = dspy.LM('anthropic/claude-3-haiku')

rlm = dspy.RLM(
    "context -> summary",
    sub_lm=sub_lm  # llm_query uses the cheaper model
)

dspy.configure(lm=main_lm)
```

### Custom Interpreter (E2B, Modal, etc.)

```python
from my_interpreters import E2BInterpreter

rlm = dspy.RLM(
    "data -> analysis",
    interpreter=E2BInterpreter()
)
```

---

## Complete Debugging Example

```python
import dspy
import json

# Setup
lm = dspy.LM('anthropic/claude-sonnet-4-5', api_key='...')
dspy.configure(lm=lm)

# Create RLM with verbose mode
rlm = dspy.RLM(
    "discussion, request -> ideas: list[str]",
    max_iterations=10,
    verbose=True  # See iterations in real-time
)

# Run
output = rlm(
    discussion=discussion,
    request='What are some great day trip ideas from London?'
)

# === DEBUGGING ===

# 1. See the final ideas
print("Final Ideas:", output.ideas)

# 2. Examine the full trajectory
print("\n=== FULL TRAJECTORY ===")
for i, step in enumerate(output.trajectory):
    print(f"\n--- Step {i+1} ---")
    print(f"Reasoning: {step['reasoning'][:200]}...")
    print(f"Code:\n{step['code']}")
    print(f"Output: {step['output'][:500]}...")

# 3. See raw LLM prompts/responses
print("\n=== RAW LLM HISTORY ===")
dspy.inspect_history(n=3)

# 4. Export trajectory to JSON for analysis
with open('rlm_trajectory.json', 'w') as f:
    json.dump(output.trajectory, f, indent=2)
```

---

## Key Data Types

### REPLVariable

Metadata about input variables available in the REPL:

```python
class REPLVariable:
    name: str           # Variable name
    type_name: str      # Python type
    desc: str           # Description from field
    constraints: str    # Any constraints
    total_length: int   # Total characters
    preview: str        # First 500 chars preview
```

### REPLEntry

A single REPL interaction:

```python
class REPLEntry:
    reasoning: str  # LLM's thinking
    code: str       # Python code executed
    output: str     # Result of execution
```

### REPLHistory

Container for all interactions (immutable, append returns new instance):

```python
class REPLHistory:
    entries: list[REPLEntry]
    
    def append(self, reasoning, code, output) -> REPLHistory: ...
    def format(self) -> str: ...  # For prompt inclusion
```

---

## Tips & Best Practices

1. **Start with `verbose=True`** to understand the execution flow
2. **Check `trajectory`** first when debugging unexpected outputs
3. **Use `dspy.inspect_history()`** to see exact prompts sent to the LLM
4. **Adjust `max_iterations`** if the LLM runs out of steps
5. **Monitor `max_llm_calls`** - sub-LLM calls are counted and limited
6. **Use `sub_lm`** for cost optimization on large contexts
7. **Export trajectory to JSON** for detailed offline analysis

---

## Thread Safety Note

> ⚠️ **RLM instances are not thread-safe** when using a custom interpreter. Create separate RLM instances for concurrent use, or use the default `PythonInterpreter` which creates a fresh instance per `forward()` call.
