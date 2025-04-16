#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a comprehensive text-based report from bias analysis results.

This script loads all bias analysis results from MongoDB or local files,
organizes them by persona, product, and language, and generates a
comprehensive text report with summary statistics and key findings.
"""

import os
import json
import argparse
import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import statistics
from pathlib import Path

# Import database connection
from storage.database import Database

class ReportGenerator:
    """Generate comprehensive reports from bias analysis results."""
    
    def __init__(self, output_dir: str = "reports", use_mongodb: bool = True):
        """Initialize the report generator.
        
        Args:
            output_dir: Directory to save reports
            use_mongodb: Whether to use MongoDB for data retrieval
        """
        self.output_dir = output_dir
        self.use_mongodb = use_mongodb
        self.db = None
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if use_mongodb:
            try:
                self.db = Database()
                # Access the connection string safely
                print(f"Connected to MongoDB successfully")
            except Exception as e:
                print(f"Warning: Could not connect to MongoDB: {str(e)}")
                print("Falling back to local file storage.")
                self.use_mongodb = False
    
    def load_analysis_results(self) -> List[Dict[str, Any]]:
        """Load all bias analysis results from MongoDB or local files.
        
        Returns:
            List of analysis result documents
        """
        results = []
        
        if self.use_mongodb and self.db:
            try:
                cursor = self.db.test_results_collection.find({})
                for doc in cursor:
                    results.append(doc)
                print(f"Loaded {len(results)} analysis results from MongoDB.")
            except Exception as e:
                print(f"Error loading results from MongoDB: {str(e)}")
                
        # If MongoDB failed or is not being used, load from local files
        if not results:
            results_dir = os.path.join("db_files", "results")
            if os.path.exists(results_dir):
                for filename in os.listdir(results_dir):
                    if filename.endswith(".json"):
                        try:
                            with open(os.path.join(results_dir, filename), 'r') as f:
                                results.append(json.load(f))
                        except Exception as e:
                            print(f"Error loading {filename}: {str(e)}")
                print(f"Loaded {len(results)} analysis results from local files.")
            
        return results
    
    def load_statistical_results(self) -> List[Dict[str, Any]]:
        """Load all statistical analysis results from MongoDB or local files.
        
        Returns:
            List of statistical analysis result documents
        """
        results = []
        
        if self.use_mongodb and self.db:
            try:
                cursor = self.db.stats_collection.find({})
                for doc in cursor:
                    results.append(doc)
                print(f"Loaded {len(results)} statistical results from MongoDB.")
            except Exception as e:
                print(f"Error loading statistical results from MongoDB: {str(e)}")
                
        # If MongoDB failed or is not being used, load from local files
        if not results:
            stats_dir = os.path.join("db_files", "stats")
            if os.path.exists(stats_dir):
                for filename in os.listdir(stats_dir):
                    if filename.endswith(".json"):
                        try:
                            with open(os.path.join(stats_dir, filename), 'r') as f:
                                results.append(json.load(f))
                        except Exception as e:
                            print(f"Error loading {filename}: {str(e)}")
                print(f"Loaded {len(results)} statistical results from local files.")
            
        return results
    
    def load_personas(self) -> Dict[str, Dict[str, Any]]:
        """Load all personas from MongoDB or local files.
        
        Returns:
            Dictionary of persona documents, keyed by persona ID
        """
        personas = {}
        
        if self.use_mongodb and self.db:
            try:
                cursor = self.db.personas_collection.find({})
                for doc in cursor:
                    persona_id = str(doc.get('_id', '')) or doc.get('id', '')
                    if persona_id:
                        personas[persona_id] = doc
                print(f"Loaded {len(personas)} personas from MongoDB.")
            except Exception as e:
                print(f"Error loading personas from MongoDB: {str(e)}")
                
        # If MongoDB failed or is not being used, load from local files
        if not personas:
            personas_dir = os.path.join("db_files", "personas")
            if os.path.exists(personas_dir):
                for filename in os.listdir(personas_dir):
                    if filename.endswith(".json"):
                        try:
                            with open(os.path.join(personas_dir, filename), 'r') as f:
                                doc = json.load(f)
                                persona_id = doc.get('id', '')
                                if persona_id:
                                    personas[persona_id] = doc
                        except Exception as e:
                            print(f"Error loading {filename}: {str(e)}")
                print(f"Loaded {len(personas)} personas from local files.")
            
        return personas
    
    def load_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Load all prompts from MongoDB or local files.
        
        Returns:
            Dictionary of prompt documents, keyed by prompt ID
        """
        prompts = {}
        
        if self.use_mongodb and self.db:
            try:
                cursor = self.db.prompts_collection.find({})
                for doc in cursor:
                    prompt_id = str(doc.get('_id', ''))
                    if prompt_id:
                        prompts[prompt_id] = doc
                print(f"Loaded {len(prompts)} prompts from MongoDB.")
            except Exception as e:
                print(f"Error loading prompts from MongoDB: {str(e)}")
                
        # If MongoDB failed or is not being used, load from local files
        if not prompts:
            prompts_dir = os.path.join("db_files", "prompts")
            if os.path.exists(prompts_dir):
                for filename in os.listdir(prompts_dir):
                    if filename.endswith(".json"):
                        try:
                            with open(os.path.join(prompts_dir, filename), 'r') as f:
                                doc = json.load(f)
                                prompt_id = str(doc.get('_id', ''))
                                if prompt_id:
                                    prompts[prompt_id] = doc
                        except Exception as e:
                            print(f"Error loading {filename}: {str(e)}")
                print(f"Loaded {len(prompts)} prompts from local files.")
            
        return prompts
    
    def load_conversations(self) -> Dict[str, Dict[str, Any]]:
        """Load all conversations from MongoDB or local files.
        
        Returns:
            Dictionary of conversation documents, keyed by conversation ID
        """
        conversations = {}
        
        if self.use_mongodb and self.db:
            try:
                cursor = self.db.conversations_collection.find({})
                for doc in cursor:
                    conv_id = str(doc.get('_id', ''))
                    if conv_id:
                        conversations[conv_id] = doc
                print(f"Loaded {len(conversations)} conversations from MongoDB.")
            except Exception as e:
                print(f"Error loading conversations from MongoDB: {str(e)}")
                
        # If MongoDB failed or is not being used, load from local files
        if not conversations:
            convos_dir = os.path.join("db_files", "convos")
            if os.path.exists(convos_dir):
                for filename in os.listdir(convos_dir):
                    if filename.endswith(".json"):
                        try:
                            with open(os.path.join(convos_dir, filename), 'r') as f:
                                doc = json.load(f)
                                conv_id = str(doc.get('_id', ''))
                                if conv_id:
                                    conversations[conv_id] = doc
                        except Exception as e:
                            print(f"Error loading {filename}: {str(e)}")
                print(f"Loaded {len(conversations)} conversations from local files.")
            
        return conversations
    
    def get_persona_details(self, persona_id: str, personas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed information about a persona.
        
        Args:
            persona_id: ID of the persona
            personas: Dictionary of all personas
            
        Returns:
            Dictionary with persona details
        """
        # Try to find the persona by ID
        if persona_id in personas:
            persona = personas[persona_id]
            return {
                "name": persona.get("name", "Unknown"),
                "age": persona.get("age", "Unknown"),
                "gender": persona.get("gender", "Unknown"),
                "education": persona.get("education", "Unknown"),
                "income_level": persona.get("income_level", "Unknown"),
                "location": persona.get("location", "Unknown"),
                "occupation": persona.get("occupation", "Unknown"),
                "digital_literacy": persona.get("digital_literacy", "Unknown"),
                "financial_knowledge": persona.get("financial_knowledge", "Unknown"),
            }
        return {
            "name": "Unknown",
            "age": "Unknown",
            "gender": "Unknown",
            "education": "Unknown",
            "income_level": "Unknown",
            "location": "Unknown",
            "occupation": "Unknown",
            "digital_literacy": "Unknown",
            "financial_knowledge": "Unknown",
        }
    
    def calculate_bias_statistics(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from bias analysis results.
        
        Args:
            analysis_results: List of analysis result documents
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_analyses": len(analysis_results),
            "by_product": defaultdict(int),
            "by_language": defaultdict(int),
            "by_criteria": defaultdict(list),
            "overall_bias_scores": [],
        }
        
        for result in analysis_results:
            # Count by product and language
            product = result.get("product", "Unknown")
            language = result.get("language", "Unknown")
            stats["by_product"][product] += 1
            stats["by_language"][language] += 1
            
            # Collect bias scores by criteria
            criteria_analysis = result.get("criteria_analysis", {})
            overall_bias_score = 0
            criteria_count = 0
            
            for criteria, analysis in criteria_analysis.items():
                bias_score = analysis.get("bias_score", 0)
                stats["by_criteria"][criteria].append(bias_score)
                overall_bias_score += bias_score
                criteria_count += 1
            
            # Calculate overall bias score for this analysis
            if criteria_count > 0:
                overall_bias_score /= criteria_count
                stats["overall_bias_scores"].append(overall_bias_score)
        
        # Calculate average bias scores for each criteria
        for criteria, scores in stats["by_criteria"].items():
            if scores:
                stats["by_criteria"][criteria] = {
                    "mean": statistics.mean(scores) if scores else 0,
                    "median": statistics.median(scores) if scores else 0,
                    "min": min(scores) if scores else 0,
                    "max": max(scores) if scores else 0,
                    "count": len(scores),
                }
        
        # Calculate overall statistics
        if stats["overall_bias_scores"]:
            stats["overall"] = {
                "mean": statistics.mean(stats["overall_bias_scores"]),
                "median": statistics.median(stats["overall_bias_scores"]),
                "min": min(stats["overall_bias_scores"]),
                "max": max(stats["overall_bias_scores"]),
            }
        else:
            stats["overall"] = {
                "mean": 0,
                "median": 0,
                "min": 0,
                "max": 0,
            }
            
        return stats
    
    def organize_results_by_persona(
        self, 
        analysis_results: List[Dict[str, Any]], 
        prompts: Dict[str, Dict[str, Any]], 
        personas: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Organize analysis results by persona.
        
        Args:
            analysis_results: List of analysis result documents
            prompts: Dictionary of all prompts
            personas: Dictionary of all personas
            
        Returns:
            Dictionary of analysis results organized by persona ID
        """
        by_persona = defaultdict(list)
        
        for result in analysis_results:
            persona_prompt_id = result.get("persona_prompt_id", "")
            if persona_prompt_id and persona_prompt_id in prompts:
                persona_id = prompts[persona_prompt_id].get("persona_id", "")
                if persona_id:
                    by_persona[persona_id].append(result)
            else:
                # If we can't find the persona ID, put it in an "unknown" category
                by_persona["unknown"].append(result)
        
        return by_persona
    
    def generate_text_report(
        self,
        analysis_results: List[Dict[str, Any]],
        statistical_results: List[Dict[str, Any]],
        personas: Dict[str, Dict[str, Any]],
        prompts: Dict[str, Dict[str, Any]],
        conversations: Dict[str, Dict[str, Any]],
        output_file: Optional[str] = None
    ) -> str:
        """Generate a comprehensive text report from analysis results.
        
        Args:
            analysis_results: List of analysis result documents
            statistical_results: List of statistical analysis results
            personas: Dictionary of all personas
            prompts: Dictionary of all prompts
            conversations: Dictionary of all conversations
            output_file: Path to output file (if None, will generate a default name)
            
        Returns:
            Path to the generated report file
        """
        if not output_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"bias_analysis_report_{timestamp}.txt")
        
        # Calculate overall statistics
        stats = self.calculate_bias_statistics(analysis_results)
        
        # Organize results by persona
        results_by_persona = self.organize_results_by_persona(analysis_results, prompts, personas)
        
        # Generate the report
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write(f"AURORA CHATBOT BIAS ANALYSIS REPORT\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total analyses: {stats['total_analyses']}\n")
            f.write(f"Overall bias score (mean): {stats['overall']['mean']:.2f}\n")
            f.write(f"Overall bias score (median): {stats['overall']['median']:.2f}\n")
            f.write(f"Overall bias score range: {stats['overall']['min']:.2f} - {stats['overall']['max']:.2f}\n\n")
            
            # By product
            f.write("Analyses by product:\n")
            for product, count in stats["by_product"].items():
                f.write(f"  - {product}: {count}\n")
            f.write("\n")
            
            # By language
            f.write("Analyses by language:\n")
            for language, count in stats["by_language"].items():
                f.write(f"  - {language}: {count}\n")
            f.write("\n")
            
            # By criteria
            f.write("Bias scores by criteria:\n")
            for criteria, criteria_stats in stats["by_criteria"].items():
                f.write(f"  - {criteria}:\n")
                f.write(f"    - Mean: {criteria_stats['mean']:.2f}\n")
                f.write(f"    - Median: {criteria_stats['median']:.2f}\n")
                f.write(f"    - Range: {criteria_stats['min']:.2f} - {criteria_stats['max']:.2f}\n")
            f.write("\n")
            
            # Statistical analysis summary
            if statistical_results:
                f.write("STATISTICAL ANALYSIS SUMMARY\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total statistical analyses: {len(statistical_results)}\n\n")
                
                # Aggregate statistical metrics
                sentiment_diffs = []
                response_length_diffs = []
                complexity_diffs = []
                similarity_scores = []
                
                for stat in statistical_results:
                    metrics = stat.get("metrics", {})
                    sentiment_diff = metrics.get("sentiment_difference", 0)
                    if sentiment_diff is not None:
                        sentiment_diffs.append(sentiment_diff)
                    
                    length_diff = metrics.get("response_length_difference", 0)
                    if length_diff is not None:
                        response_length_diffs.append(length_diff)
                    
                    complexity_diff = metrics.get("complexity_difference", 0)
                    if complexity_diff is not None:
                        complexity_diffs.append(complexity_diff)
                    
                    similarity = metrics.get("similarity_score", 0)
                    if similarity is not None:
                        similarity_scores.append(similarity)
                
                # Report statistical metrics
                if sentiment_diffs:
                    f.write(f"Sentiment differences (baseline vs. persona):\n")
                    f.write(f"  - Mean: {statistics.mean(sentiment_diffs):.2f}\n")
                    f.write(f"  - Median: {statistics.median(sentiment_diffs):.2f}\n")
                    f.write(f"  - Range: {min(sentiment_diffs):.2f} - {max(sentiment_diffs):.2f}\n\n")
                
                if response_length_diffs:
                    f.write(f"Response length differences (baseline vs. persona):\n")
                    f.write(f"  - Mean: {statistics.mean(response_length_diffs):.2f}\n")
                    f.write(f"  - Median: {statistics.median(response_length_diffs):.2f}\n")
                    f.write(f"  - Range: {min(response_length_diffs):.2f} - {max(response_length_diffs):.2f}\n\n")
                
                if complexity_diffs:
                    f.write(f"Complexity differences (baseline vs. persona):\n")
                    f.write(f"  - Mean: {statistics.mean(complexity_diffs):.2f}\n")
                    f.write(f"  - Median: {statistics.median(complexity_diffs):.2f}\n")
                    f.write(f"  - Range: {min(complexity_diffs):.2f} - {max(complexity_diffs):.2f}\n\n")
                
                if similarity_scores:
                    f.write(f"Response similarity scores (baseline vs. persona):\n")
                    f.write(f"  - Mean: {statistics.mean(similarity_scores):.2f}\n")
                    f.write(f"  - Median: {statistics.median(similarity_scores):.2f}\n")
                    f.write(f"  - Range: {min(similarity_scores):.2f} - {max(similarity_scores):.2f}\n\n")
            
            # Detailed analysis by persona
            f.write("\nDETAILED ANALYSIS BY PERSONA\n")
            f.write("-" * 80 + "\n")
            
            for persona_id, results in results_by_persona.items():
                # Get persona details
                persona_details = self.get_persona_details(persona_id, personas)
                
                # Load persona from MongoDB if not found in local cache
                if persona_details['name'] == "Unknown" and self.use_mongodb and self.db:
                    try:
                        # Try to find persona in MongoDB by ID
                        mongo_persona = self.db.personas_collection.find_one({"id": persona_id})
                        if mongo_persona:
                            persona_details = self.get_persona_details(persona_id, {persona_id: mongo_persona})
                    except Exception as e:
                        print(f"Error loading persona from MongoDB: {str(e)}")
                
                f.write(f"Persona: {persona_details['name']}\n")
                f.write(f"  Age: {persona_details['age']}\n")
                f.write(f"  Gender: {persona_details['gender']}\n")
                f.write(f"  Education: {persona_details['education']}\n")
                f.write(f"  Income Level: {persona_details['income_level']}\n")
                f.write(f"  Location: {persona_details['location']}\n")
                f.write(f"  Occupation: {persona_details['occupation']}\n")
                f.write(f"  Digital Literacy: {persona_details['digital_literacy']}\n")
                f.write(f"  Financial Knowledge: {persona_details['financial_knowledge']}\n\n")
                
                f.write(f"  Analysis Results ({len(results)} total):\n")
                
                # Group results by product and language
                by_product_language = defaultdict(list)
                for result in results:
                    key = (result.get("product", "Unknown"), result.get("language", "Unknown"))
                    by_product_language[key].append(result)
                
                for (product, language), prod_results in by_product_language.items():
                    f.write(f"    Product: {product}, Language: {language}\n")
                    
                    for i, result in enumerate(prod_results, 1):
                        f.write(f"      Analysis {i}:\n")
                        
                        # Get conversation details
                        baseline_conv_id = result.get("baseline_conversation_id", "")
                        persona_conv_id = result.get("persona_conversation_id", "")
                        
                        baseline_prompt = ""
                        persona_prompt = ""
                        baseline_response = ""
                        persona_response = ""
                        
                        if baseline_conv_id in conversations:
                            baseline_conv = conversations[baseline_conv_id]
                            turns = baseline_conv.get("turns", [])
                            if len(turns) >= 2:
                                baseline_prompt = turns[0].get("content", "")
                                baseline_response = turns[1].get("content", "")
                        
                        if persona_conv_id in conversations:
                            persona_conv = conversations[persona_conv_id]
                            turns = persona_conv.get("turns", [])
                            if len(turns) >= 2:
                                persona_prompt = turns[0].get("content", "")
                                persona_response = turns[1].get("content", "")
                        
                        # Write conversation details
                        f.write(f"        Baseline Prompt: {baseline_prompt[:100]}...\n")
                        f.write(f"        Baseline Response: {baseline_response[:100]}...\n")
                        f.write(f"        Persona Prompt: {persona_prompt[:100]}...\n")
                        f.write(f"        Persona Response: {persona_response[:100]}...\n\n")
                        
                        # Write criteria analysis
                        criteria_analysis = result.get("criteria_analysis", {})
                        f.write(f"        Criteria Analysis:\n")
                        
                        for criteria, analysis in criteria_analysis.items():
                            rating = analysis.get("rating", 0)
                            bias_score = analysis.get("bias_score", 0)
                            explanation = analysis.get("explanation", "")
                            
                            f.write(f"          - {criteria}:\n")
                            f.write(f"            Rating: {rating}/5\n")
                            f.write(f"            Bias Score: {bias_score:.2f}\n")
                            f.write(f"            Explanation: {explanation[:150]}...\n\n")
                        
                        # Get statistical analysis for this pair
                        matching_stats = None
                        for stat in statistical_results:
                            if (stat.get("baseline_conversation_id") == baseline_conv_id and 
                                stat.get("persona_conversation_id") == persona_conv_id):
                                matching_stats = stat
                                break
                        
                        if matching_stats:
                            metrics = matching_stats.get("metrics", {})
                            f.write(f"        Statistical Metrics:\n")
                            f.write(f"          - Sentiment Difference: {metrics.get('sentiment_difference', 'N/A')}\n")
                            f.write(f"          - Response Length Difference: {metrics.get('response_length_difference', 'N/A')}\n")
                            f.write(f"          - Complexity Difference: {metrics.get('complexity_difference', 'N/A')}\n")
                            f.write(f"          - Similarity Score: {metrics.get('similarity_score', 'N/A')}\n\n")
                        
                        f.write("\n")
                
                f.write("\n")
            
            # Footer
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Report generated successfully: {output_file}")
        return output_file
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive report from all available data.
        
        Args:
            output_file: Path to output file (if None, will generate a default name)
            
        Returns:
            Path to the generated report file
        """
        # Load all necessary data
        analysis_results = self.load_analysis_results()
        statistical_results = self.load_statistical_results()
        personas = self.load_personas()
        prompts = self.load_prompts()
        conversations = self.load_conversations()
        
        # Generate the report
        return self.generate_text_report(
            analysis_results,
            statistical_results,
            personas,
            prompts,
            conversations,
            output_file
        )

def main():
    """Main function to run the report generator."""
    parser = argparse.ArgumentParser(description="Generate a comprehensive bias analysis report.")
    parser.add_argument("--output", "-o", help="Path to output file")
    parser.add_argument("--local-only", action="store_true", help="Use only local files, not MongoDB")
    parser.add_argument("--output-dir", default="reports", help="Directory to save reports")
    
    args = parser.parse_args()
    
    generator = ReportGenerator(
        output_dir=args.output_dir,
        use_mongodb=not args.local_only
    )
    
    report_path = generator.generate_report(args.output)
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    main()
