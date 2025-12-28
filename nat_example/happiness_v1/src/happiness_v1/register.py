import json
import os
from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

# Import our standalone tools
from happiness_v1.utils.happiness_stats import load_data, get_country_stats, get_year_stats

# 1. Input schemas: tell LLM what each tool expects
class CountryStatsInput(BaseModel):
    country: str = Field(
        description="Country name to filter the Happiness Report data. For example: 'Finland', 'United States', 'India'."
    )

class YearStatsInput(BaseModel):
    year: int = Field(
        description="Year to filter the Happiness Report data. For example: 2019, 2020, 2021."
    )

# 2. Create configs 
class CountryStatsConfig(FunctionBaseConfig, name="country_stats"):  
    """Configuration for calculating country-specific happiness statistics."""
    pass

class YearStatsConfig(FunctionBaseConfig, name="year_stats"):  
    """Configuration for calculating year-specific happiness statistics."""
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
        input_schema=CountryStatsInput
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
        input_schema=YearStatsInput
    )