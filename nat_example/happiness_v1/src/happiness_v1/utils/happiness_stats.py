import pandas as pd
import json
import os
from typing import Dict, List, Optional, Union

def load_data() -> pd.DataFrame:
    """ArithmeticError
        Load and preprocess the World Happiness Report data.
        Returns:
            pd.DataFrame: Preprocessed DataFrame containing happiness data from 2019 onwards.
        """
    df = pd.read_excel('/Users/marie/Documents/github/miptgirl_medium/nat_example/whr2025_data.xlsx')
    df = df[df.Year >= 2019]
    df = df.drop(['Lower whisker', 'Upper whisker'], axis=1)
    df.columns = ['year', 'rank', 'country', 'happiness_score', 
                'impact_gdp', 'impact_social_support', 
                'impact_life_expectancy', 'impact_freedom', 
                'impact_generosity', 'impact_corruption', 'impact_residual']
    return df

def get_country_stats(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Get happiness statistics for a specific country.
    
    Args:
        df (pd.DataFrame): DataFrame containing happiness data.
        country (str): Name of the country to filter by.
        
    Returns:
        pd.DataFrame: Filtered DataFrame with statistics for the specified country.
    """
    return df[df['country'].str.contains(country, case=False)]

def get_year_stats(df: pd.DataFrame, year: int) -> str:
    """
    Get happiness statistics for a specific year.
    
    Args:
        df (pd.DataFrame): DataFrame containing happiness data.
        year (int): Year to filter by.
        
    Returns:
        summary (str): Summary statistics for the specified year.
    """

    year_df = df[df['year'] == year].sort_values('rank')
    top5_countries = f'''
    Top 5 Countries in {year} by Happiness Rank:
    {year_df.head(5)[["rank", "country", "happiness_score"]].to_string(index=False)}
    '''

    bottom5_countries = f'''
    Bottom 5 Countries in {year} by Happiness Rank:
    {year_df.tail(5)[["rank", "country", "happiness_score"]].to_string(index=False)}
    '''

    scores_mean = f'''
    Average Happiness Score in {year}: 
    {year_df[['happiness_score', 'impact_gdp', 'impact_social_support', 
         'impact_life_expectancy', 'impact_freedom', 
         'impact_generosity', 'impact_corruption']].mean().to_string()}
    '''

    return  top5_countries + '\n' + bottom5_countries + '\n' + scores_mean


