"""
Premium cost calculator for materials

This module provides a calculator to estimate the premium costs associated with
renewable energy and sustainable material sourcing in the battery supply chain.
It calculates costs based on material types, quantities, and country of origin.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import json
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaterialPremiumCostCalculator:
    """
    Calculator for premium costs associated with materials based on tier, material type, and country.
    
    This class loads premium cost data from a configuration file and calculates
    the expected premium costs for materials in both original and optimized scenarios.
    It provides methods to compare costs and calculate cost reductions.
    
    Attributes:
        cost_data (Dict): Dictionary containing premium cost data by tier, material, and country
        stable_var_dir (str): Directory path for stable variables including cost_by_tier.json
        user_id (str, optional): User identifier for user-specific data
    """
    
    def __init__(self, 
                 stable_var_dir: str = "stable_var",
                 user_id: Optional[str] = None,
                 debug_mode: bool = False):
        """
        Initialize the MaterialPremiumCostCalculator.
        
        Args:
            stable_var_dir: Path to the directory containing cost_by_tier.json
            user_id: Optional user ID for user-specific data
            debug_mode: Whether to enable debug logging
        """
        self.debug_mode = debug_mode
        self.user_id = user_id
        self.stable_var_dir = stable_var_dir
        self.cost_data = {}
        
        # Set up logging level based on debug mode
        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
        
        # Load premium cost data
        self._load_cost_data()
        
        # Debug output
        if self.debug_mode:
            self._print_debug_info()
    
    def _load_cost_data(self) -> None:
        """
        Load premium cost data from cost_by_tier.json.
        
        The function first attempts to load user-specific cost data if a user_id is provided.
        If that fails or no user_id is provided, it falls back to the standard cost_by_tier.json file.
        
        Raises:
            FileNotFoundError: If the cost data file cannot be found
            json.JSONDecodeError: If the cost data file contains invalid JSON
        """
        try:
            # Determine the file path based on user_id
            if self.user_id:
                # Try user-specific file first
                file_path = Path(self.stable_var_dir) / self.user_id / "cost_by_tier.json"
                if not file_path.exists():
                    # Fall back to standard file
                    file_path = Path(self.stable_var_dir) / "cost_by_tier.json"
            else:
                file_path = Path(self.stable_var_dir) / "cost_by_tier.json"
            
            # Load and parse the JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Process the tier data into a more accessible structure
            self.cost_data = {
                'tier_data': data.get('tier_data', []),
                'unit_info': data.get('unit_info', {})
            }
            
            # Log success
            logger.debug(f"Successfully loaded cost data from {file_path}")
            
        except FileNotFoundError:
            logger.error(f"Cost data file not found at {file_path}")
            raise FileNotFoundError(f"Cost data file not found at {file_path}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in cost data file at {file_path}")
            raise json.JSONDecodeError(f"Invalid JSON in cost data file at {file_path}", "", 0)
    
    def _print_debug_info(self) -> None:
        """Print debug information about loaded cost data."""
        print("===== MaterialPremiumCostCalculator Initialization =====")
        print(f"• Loaded cost data for {len(self.cost_data['tier_data'])} material-tier-country combinations")
        
        # Count unique materials, tiers, and countries
        materials = set(item['material'] for item in self.cost_data['tier_data'])
        tiers = set(item['tier'] for item in self.cost_data['tier_data'])
        countries = set(item['country'] for item in self.cost_data['tier_data'])
        
        print(f"• Unique materials: {len(materials)}")
        print(f"• Unique tiers: {len(tiers)}")
        print(f"• Unique countries: {len(countries)}")
        
        # Show unit information
        if 'unit_info' in self.cost_data:
            print(f"\nUnit Information:")
            for key, value in self.cost_data['unit_info'].items():
                print(f"• {key}: {value}")
    
    def find_material_cost(self, material: str, country: str, tier: str = "Tier1") -> float:
        """
        Find the premium cost for a specific material, country, and tier.
        
        Args:
            material: Material name
            country: Country of origin
            tier: Tier level (default: "Tier1")
            
        Returns:
            float: The expected premium cost for the material
            
        Raises:
            ValueError: If no matching material cost data is found
        """
        # Search for matching cost data
        for item in self.cost_data['tier_data']:
            if (item['material'] == material and 
                item['country'] == country and 
                item['tier'] == tier):
                return item['expected_cost']
        
        # If no exact match, try to find a similar material
        for item in self.cost_data['tier_data']:
            if (material.lower() in item['material'].lower() and 
                item['country'] == country and 
                item['tier'] == tier):
                logger.debug(f"Using similar material match: {material} -> {item['material']}")
                return item['expected_cost']
        
        # If no match found, log warning and return 0
        logger.warning(f"No premium cost data found for material={material}, country={country}, tier={tier}")
        return 0.0
    
    def calculate_baseline_premium_costs(self, materials_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate baseline premium costs using original material data.
        
        Args:
            materials_data: DataFrame containing material information
                           Required columns: '자재명', '자재품목', '국가', '제품총소요량(kg)'
                           
        Returns:
            Dict: Dictionary containing the calculated premium costs
        """
        if not isinstance(materials_data, pd.DataFrame):
            raise TypeError("materials_data must be a pandas DataFrame")
        
        # Check for required columns
        required_columns = ['자재명', '자재품목', '국가', '제품총소요량(kg)']
        for column in required_columns:
            if column not in materials_data.columns:
                raise ValueError(f"Required column '{column}' not found in materials_data")
        
        total_premium_cost = 0.0
        material_costs = []
        
        # Calculate premium cost for each material
        for _, row in materials_data.iterrows():
            material_name = row['자재명']
            material_category = row['자재품목']
            country = row['국가']
            quantity = row['제품총소요량(kg)']
            
            # Find the premium cost for this material
            unit_premium_cost = self.find_material_cost(material_category, country)
            
            # Calculate the total premium cost for this material
            material_premium_cost = unit_premium_cost * quantity
            
            # Add to total
            total_premium_cost += material_premium_cost
            
            # Store individual material cost
            material_costs.append({
                'material_name': material_name,
                'material_category': material_category,
                'country': country,
                'quantity': quantity,
                'unit_premium_cost': unit_premium_cost,
                'total_premium_cost': material_premium_cost
            })
        
        return {
            'total_premium_cost': total_premium_cost,
            'material_costs': material_costs
        }
    
    def calculate_optimized_premium_costs(self, 
                                         original_materials_data: pd.DataFrame,
                                         optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimized premium costs after optimization.
        
        Args:
            original_materials_data: DataFrame containing original material information
            optimization_result: Optimization result dictionary from MaterialBasedOptimizer
            
        Returns:
            Dict: Dictionary containing the calculated optimized premium costs
        """
        if not isinstance(original_materials_data, pd.DataFrame):
            raise TypeError("original_materials_data must be a pandas DataFrame")
        
        if not isinstance(optimization_result, dict) or 'variables' not in optimization_result:
            raise ValueError("optimization_result must be a dictionary containing 'variables' key")
        
        # Check for required columns in materials data
        required_columns = ['자재명', '자재품목', '국가', '제품총소요량(kg)']
        for column in required_columns:
            if column not in original_materials_data.columns:
                raise ValueError(f"Required column '{column}' not found in original_materials_data")
        
        variables = optimization_result['variables']
        total_premium_cost = 0.0
        material_costs = []
        
        # Calculate premium cost for each material
        for _, row in original_materials_data.iterrows():
            material_name = row['자재명']
            material_category = row['자재품목']
            country = row['국가']
            quantity = row['제품총소요량(kg)']
            
            # Extract the optimized RE application rates if available
            tier1_re = 0.0
            tier2_re = 0.0
            
            # Look for the material's RE rates in the optimization results
            tier1_key = f'tier1_re_{material_name}'
            tier2_key = f'tier2_re_{material_name}'
            
            if tier1_key in variables:
                tier1_re = variables[tier1_key]
            if tier2_key in variables:
                tier2_re = variables[tier2_key]
            
            # Find the base premium costs for this material
            tier1_premium_cost = self.find_material_cost(material_category, country, "Tier1")
            tier2_premium_cost = self.find_material_cost(material_category, country, "Tier2")
            
            # Calculate the weighted premium cost based on RE application rates
            weighted_premium_cost = (tier1_premium_cost * tier1_re) + (tier2_premium_cost * tier2_re)
            
            # Calculate the total premium cost for this material
            material_premium_cost = weighted_premium_cost * quantity
            
            # Add to total
            total_premium_cost += material_premium_cost
            
            # Store individual material cost
            material_costs.append({
                'material_name': material_name,
                'material_category': material_category,
                'country': country,
                'quantity': quantity,
                'tier1_re': tier1_re,
                'tier2_re': tier2_re,
                'weighted_premium_cost': weighted_premium_cost,
                'total_premium_cost': material_premium_cost
            })
        
        return {
            'total_premium_cost': total_premium_cost,
            'material_costs': material_costs
        }
    
    def calculate_cost_reduction(self, baseline_costs: Dict[str, Any], optimized_costs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate cost reduction percentage between baseline and optimized costs.
        
        Args:
            baseline_costs: Baseline premium costs dictionary
            optimized_costs: Optimized premium costs dictionary
            
        Returns:
            Dict: Dictionary containing the cost reduction information
        """
        # Validate inputs
        if not isinstance(baseline_costs, dict) or 'total_premium_cost' not in baseline_costs:
            raise ValueError("baseline_costs must be a dictionary containing 'total_premium_cost' key")
        
        if not isinstance(optimized_costs, dict) or 'total_premium_cost' not in optimized_costs:
            raise ValueError("optimized_costs must be a dictionary containing 'total_premium_cost' key")
        
        # Extract total costs
        baseline_total = baseline_costs['total_premium_cost']
        optimized_total = optimized_costs['total_premium_cost']
        
        # Calculate absolute and percentage reduction
        absolute_reduction = baseline_total - optimized_total
        
        # Handle division by zero
        if baseline_total == 0:
            percentage_reduction = 0.0
            logger.warning("Baseline total premium cost is zero, cannot calculate percentage reduction")
        else:
            percentage_reduction = (absolute_reduction / baseline_total) * 100
        
        return {
            'baseline_cost': baseline_total,
            'optimized_cost': optimized_total,
            'absolute_reduction': absolute_reduction,
            'percentage_reduction': percentage_reduction
        }
    
    def get_formatted_results(self, 
                              baseline_costs: Dict[str, Any], 
                              optimized_costs: Dict[str, Any],
                              reduction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the cost calculation results in a user-friendly format.
        
        Args:
            baseline_costs: Baseline premium costs dictionary
            optimized_costs: Optimized premium costs dictionary
            reduction: Cost reduction dictionary
            
        Returns:
            Dict: Formatted results dictionary
        """
        formatted = {
            'baseline_premium_cost': f"${baseline_costs['total_premium_cost']:.4f}",
            'optimized_premium_cost': f"${optimized_costs['total_premium_cost']:.4f}",
            'absolute_cost_reduction': f"${reduction['absolute_reduction']:.4f}",
            'percentage_cost_reduction': f"{reduction['percentage_reduction']:.2f}%"
        }
        
        # Add material level details
        materials_data = []
        
        # Process baseline material costs
        baseline_materials = {item['material_name']: item for item in baseline_costs['material_costs']}
        optimized_materials = {item['material_name']: item for item in optimized_costs['material_costs']}
        
        # Combine material data
        all_materials = set(baseline_materials.keys()) | set(optimized_materials.keys())
        
        for material_name in all_materials:
            material_data = {
                'name': material_name
            }
            
            # Add baseline data if available
            if material_name in baseline_materials:
                baseline = baseline_materials[material_name]
                material_data.update({
                    'category': baseline['material_category'],
                    'country': baseline['country'],
                    'quantity': f"{baseline['quantity']:.2f} kg",
                    'baseline_unit_cost': f"${baseline['unit_premium_cost']:.4f}/kg",
                    'baseline_total_cost': f"${baseline['total_premium_cost']:.4f}"
                })
            
            # Add optimized data if available
            if material_name in optimized_materials:
                optimized = optimized_materials[material_name]
                material_data.update({
                    'tier1_re': f"{optimized['tier1_re']*100:.1f}%",
                    'tier2_re': f"{optimized['tier2_re']*100:.1f}%",
                    'optimized_unit_cost': f"${optimized['weighted_premium_cost']:.4f}/kg",
                    'optimized_total_cost': f"${optimized['total_premium_cost']:.4f}"
                })
                
                # Calculate material-level reduction
                if material_name in baseline_materials:
                    baseline_cost = baseline_materials[material_name]['total_premium_cost']
                    optimized_cost = optimized['total_premium_cost']
                    abs_reduction = baseline_cost - optimized_cost
                    
                    if baseline_cost > 0:
                        pct_reduction = (abs_reduction / baseline_cost) * 100
                    else:
                        pct_reduction = 0.0
                    
                    material_data.update({
                        'absolute_reduction': f"${abs_reduction:.4f}",
                        'percentage_reduction': f"{pct_reduction:.2f}%"
                    })
            
            materials_data.append(material_data)
        
        formatted['materials'] = materials_data
        return formatted


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Create a calculator
    calculator = MaterialPremiumCostCalculator(debug_mode=True)
    
    # Sample material data
    materials_data = pd.DataFrame({
        '자재명': ['Material1', 'Material2'],
        '자재품목': ['양극재', '음극재(천연)'],
        '국가': ['한국', '중국'],
        '제품총소요량(kg)': [100, 200]
    })
    
    # Calculate baseline costs
    baseline_costs = calculator.calculate_baseline_premium_costs(materials_data)
    
    # Sample optimization result
    optimization_result = {
        'variables': {
            'tier1_re_Material1': 0.8,
            'tier2_re_Material1': 0.5,
            'tier1_re_Material2': 0.7,
            'tier2_re_Material2': 0.4
        }
    }
    
    # Calculate optimized costs
    optimized_costs = calculator.calculate_optimized_premium_costs(
        materials_data, optimization_result
    )
    
    # Calculate cost reduction
    reduction = calculator.calculate_cost_reduction(baseline_costs, optimized_costs)
    
    # Get formatted results
    formatted_results = calculator.get_formatted_results(
        baseline_costs, optimized_costs, reduction
    )
    
    # Print results
    import json
    print(json.dumps(formatted_results, indent=2))