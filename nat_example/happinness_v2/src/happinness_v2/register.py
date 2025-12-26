import json
import os
from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.builder.framework_enum import LLMFrameworkEnum

# Import our standalone tools
from happinness_v2.utils.happiness_stats import load_data, get_country_stats, get_year_stats

# Import calculator agent
from happinness_v2.utils.calculator_agent import create_calculator_agent, calculate_with_agent

# 1. Input schemas: tell LLM what each tool expects
class CountryStatsInput(BaseModel):
    country: str = Field(
        description="Country name to filter the Happiness Report data. For example: 'Finland', 'United States', 'India'."
    )

class YearStatsInput(BaseModel):
    year: int = Field(
        description="Year to filter the Happiness Report data. For example: 2019, 2020, 2021."
    )

class CalculatorInput(BaseModel):
    question: str = Field(
        description="Question related to maths or calculations needed for happiness statistics."
    )


# 2. Create configs 
class CountryStatsConfig(FunctionBaseConfig, name="country_stats"):  
    """Configuration for calculating country-specific happiness statistics."""
    pass

class YearStatsConfig(FunctionBaseConfig, name="year_stats"):  
    """Configuration for calculating year-specific happiness statistics."""
    pass

class CalculatorAgentConfig(FunctionBaseConfig, name="calculator_agent"):
    """Configuration for the mathematical calculator agent."""
    pass

# 3. Register functions
@register_function(config_type=CountryStatsConfig)
async def country_stats_tool(config: CountryStatsConfig, builder: Builder):
    """Register tool for calculating country-specific happiness statistics."""
    df = load_data()

    async def _wrapper(country: str) -> str:
        result = get_country_stats(df, country)
        return result

    yield FunctionInfo.from_fn(
        _wrapper,
        input_schema=CountryStatsInput,
        description="Get happiness statistics for a specific country from the World Happiness Report data."
    )

@register_function(config_type=YearStatsConfig)
async def year_stats_tool(config: YearStatsConfig, builder: Builder):
    """Register tool for calculating year-specific happiness statistics."""
    df = load_data()

    async def _wrapper(year: int) -> str:
        result = get_year_stats(df, year)
        return result

    yield FunctionInfo.from_fn(
        _wrapper,
        input_schema=YearStatsInput,
        description="Get happiness statistics for a specific year from the World Happiness Report data."
    )

@register_function(config_type=CalculatorAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def calculator_agent_tool(config: CalculatorAgentConfig, builder: Builder):
    """Register the LangGraph calculator agent as a NAT tool."""
    
    llm = await builder.get_llm("calculator_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    calculator_agent = create_calculator_agent(llm)
    
    async def _wrapper(question: str) -> str:
        # Use the calculator agent to process the question
        result = calculate_with_agent(question, calculator_agent)
        
        # Format the response as a JSON string
        response = {
            "calculation_steps": result["steps"],
            "final_result": result["final_result"],
            "explanation": result["explanation"]
        }
        return json.dumps(response, indent=2)
    
    yield FunctionInfo.from_fn(
        _wrapper,
        input_schema=CalculatorInput,
        description="Perform complex mathematical calculations using a calculator agent."
    )
