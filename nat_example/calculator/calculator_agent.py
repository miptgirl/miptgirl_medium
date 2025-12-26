"""
Simple Calculator Agent using LangGraph

A mathematical calculation agent that uses Python code evaluation for all calculations.
Single tool approach for simplicity and flexibility.

AVAILABLE TOOL:
---------------
1. evaluate_python(code: str) -> str
   Description: Execute Python code for mathematical calculations
   Supports: Any valid Python mathematical expression or code
   Example: evaluate_python('import math; math.sqrt(144) + 2**3')
   Use for: All mathematical calculations

EXAMPLE USAGE:
--------------
from calculator_agent import calculate

result = calculate("What is the square root of 144 plus 2 to the power of 3?")
print(result["explanation"])
"""

import os
import re
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
import operator


# State definition
class CalculatorState(TypedDict):
    """State for the calculator agent"""
    messages: Annotated[list[BaseMessage], operator.add]
    calculation_steps: list[str]
    final_result: float | None


@tool
def evaluate_python(code: str) -> str:
    """
    Execute Python code for mathematical calculations.
    
    Args:
        code: Python code to execute. Can include imports like math, numpy.
              Example: "import math; result = math.sqrt(144) + 2**3; result"
    
    Returns:
        The result of the calculation as a string
    """
    try:
        # Create a safe namespace with common math functions
        import math
        safe_globals = {
            "__builtins__": {},
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "len": len,
        }
        
        # Try to import numpy if available
        try:
            import numpy as np
            safe_globals["np"] = np
            safe_globals["numpy"] = np
        except ImportError:
            pass
        
        local_vars = {}
        
        # Execute the code
        exec(code, safe_globals, local_vars)
        
        # Try to get 'result' variable, or evaluate last expression
        if 'result' in local_vars:
            return str(local_vars['result'])
        
        # If no result variable, try to evaluate the code as an expression
        try:
            return str(eval(code, safe_globals, local_vars))
        except:
            # Return the last assigned variable if any
            if local_vars:
                last_value = list(local_vars.values())[-1]
                return str(last_value)
            return "Code executed successfully but no result returned"
            
    except Exception as e:
        return f"Error: {str(e)}"


# System prompt
SYSTEM_PROMPT = """You are a mathematical calculation assistant.

When given a calculation request:
1. Break it down into clear steps if needed
2. Use the evaluate_python tool for all calculations
3. Show your work clearly
4. Provide a final answer with explanation

Available tool:
- evaluate_python: Execute Python code for any mathematical calculation.
  You can use math module functions (sqrt, sin, cos, log, etc.) and numpy if available.
  Always assign the final result to a variable called 'result'.
  
Example usage:
  evaluate_python("import math; result = math.sqrt(144) + 2**3")
  evaluate_python("result = (100 - 20) / 4 * 3")
  evaluate_python("import math; result = math.log(100, 10)")

Always think step by step and use the tool for calculations."""


def create_calculator_agent(llm=None):
    """Create the calculator agent graph"""
    
    # Initialize LLM if not provided
    if llm is None:
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.0,
            max_tokens=1024
        )
    
    # Bind tools to LLM
    tools = [evaluate_python]
    llm_with_tools = llm.bind_tools(tools)
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    def agent(state: CalculatorState) -> dict:
        """Main agent function"""
        messages = state["messages"]
        
        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        
        # Update calculation steps if we have tool calls
        calculation_steps = state.get("calculation_steps", [])
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                step_desc = f"evaluate_python({tool_call['args'].get('code', '')})"
                calculation_steps.append(step_desc)
        
        return {
            "messages": [response],
            "calculation_steps": calculation_steps
        }
    
    def should_continue(state: CalculatorState) -> Literal["tools", "end"]:
        """Decide whether to continue"""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "end"
    
    # Create graph
    graph = StateGraph(CalculatorState)
    
    # Add nodes
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    
    # Add edges
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    graph.add_edge("tools", "agent")
    
    return graph.compile()


def calculate(question: str) -> dict:
    """
    Main entry point for calculations.
    
    Args:
        question: A mathematical question or calculation request
    
    Returns:
        Dictionary with result, steps, and explanation
    """
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "calculation_steps": [],
        "final_result": None
    }
    
    calculator_agent = create_calculator_agent()
    result = calculator_agent.invoke(initial_state)
    
    # Extract final result from steps
    final_result = None
    explanation = ""
    
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            explanation = msg.content
            # Look for numbers in the final answer
            numbers = re.findall(r'-?\d+\.?\d*', msg.content)
            if numbers:
                try:
                    final_result = float(numbers[-1])
                except:
                    pass
            break
    
    return {
        "question": question,
        "steps": result.get("calculation_steps", []),
        "final_result": final_result,
        "explanation": explanation
    }


if __name__ == "__main__":
    # Test the calculator
    test_questions = [
        "What is 25 * 4 + 100 / 5?",
        "Calculate the square root of 144 plus 2 to the power of 3",
        "What is the compound annual growth rate if a value went from 100 to 150 over 5 years?",
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        result = calculate(question)
        print(f"Steps: {result['steps']}")
        print(f"Final Result: {result['final_result']}")
        print(f"Explanation: {result['explanation'][:300]}...")
