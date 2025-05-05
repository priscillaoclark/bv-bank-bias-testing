#!/usr/bin/env python
"""
Run Bias Test for Aurora Chatbot

This script orchestrates the entire bias testing process:
1. Generate prompts (both baseline and persona-specific)
2. Send prompts to the Aurora chatbot
3. Run statistical analysis on conversation pairs
4. Analyze responses for bias using both statistical and qualitative approaches

This version uses only local JSON files for storage (no MongoDB dependency).

Usage:
    python run_bias_test.py --products 2 --personas 3
"""

import os
import sys
import json
import argparse
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from storage.json_database import JSONDatabase as Database
from dotenv import load_dotenv

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from persona_generator import PersonaGenerator
from prompt_generator import PromptGenerator
from chatbot_tester import LightweightChatbotTester
from bias_analyzer import BiasAnalyzer
from statistical_bias_analyzer import StatisticalBiasAnalyzer
from utils.logging_config import get_run_logger

# Load environment variables
load_dotenv()

class BiasTester:
    """Orchestrate the entire bias testing process."""
    
    def __init__(self, logger=None):
        """Initialize the BiasTester.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.persona_generator = PersonaGenerator()
        self.prompt_generator = PromptGenerator()
        self.statistical_analyzer = StatisticalBiasAnalyzer()
        self.bias_analyzer = BiasAnalyzer()
        
        # Track newly generated items for this run
        self.new_persona_ids = []
        self.new_prompt_ids = []
        self.new_conversation_ids = {}
    
    def generate_personas(self, num_personas: int, diversity_strategy: str = "mixed", temperature: float = None, enforce_diversity: bool = True) -> List[str]:
        """Generate personas for testing.
        
        Args:
            num_personas: Number of personas to generate
            diversity_strategy: Strategy for temperature diversity
            temperature: Specific temperature to use (overrides diversity_strategy)
            enforce_diversity: Whether to enforce diversity between generated personas
        """
        self.logger.info(f"Generating {num_personas} personas with '{diversity_strategy}' diversity strategy")
        if enforce_diversity:
            self.logger.info("Diversity validation is ENABLED - personas will be checked for uniqueness")
        else:
            self.logger.info("Diversity validation is DISABLED - personas may have similar characteristics")
        
        if temperature is not None:
            self.logger.info(f"Using fixed temperature: {temperature}")
            # Generate personas with a specific temperature
            if enforce_diversity:
                # Use the new ensure_diverse_persona method for each persona
                persona_ids = []
                generated_personas = []
                existing_personas = []
                
                # Try to get existing personas from the database for comparison
                try:
                    if hasattr(self.persona_generator, 'db') and self.persona_generator.db:
                        existing_personas = self.persona_generator.db.get_all_personas()
                        print(f"Retrieved {len(existing_personas)} existing personas for diversity validation")
                except Exception as e:
                    print(f"Warning: Could not retrieve existing personas: {str(e)}")
                
                for i in range(num_personas):
                    print(f"\nGenerating persona {i+1}/{num_personas}...")
                    # Combine existing and newly generated personas for comparison
                    all_personas_for_comparison = existing_personas + [p for p in generated_personas if isinstance(p, dict)]
                    
                    # Generate a diverse persona
                    persona = self.persona_generator.ensure_diverse_persona(
                        all_personas_for_comparison,
                        temperature=temperature,
                        max_attempts=3
                    )
                    
                    # Save the persona and add to our list
                    persona_id = self.persona_generator.save_persona(persona)
                    if persona_id:
                        persona_ids.append(persona_id)
                        generated_personas.append(persona)  # Add to our list of generated personas
                
                return persona_ids
            else:
                # Original implementation without diversity validation
                persona_ids = []
                for i in range(num_personas):
                    print(f"\nGenerating persona {i+1}/{num_personas}...")
                    persona = self.persona_generator.generate_persona(temperature=temperature)
                    persona_id = self.persona_generator.save_persona(persona)
                    if persona_id:
                        persona_ids.append(persona_id)
                return persona_ids
        else:
            # Generate personas with the specified diversity strategy using the updated method
            return self.persona_generator.generate_personas(
                num_personas, 
                diversity_strategy=diversity_strategy,
                enforce_diversity=enforce_diversity
            )
    
    def generate_prompts(self, num_products: int) -> List[str]:
        """Generate prompts for testing."""
        self.logger.info(f"Generating prompts for {num_products} products")
        return self.prompt_generator.generate_prompts_for_personas(num_products)
    
    def test_prompts(self, prompt_ids: List[str]) -> Dict[str, str]:
        """Test prompts with the Aurora chatbot."""
        self.logger.info("Testing prompts with Aurora chatbot")
        conversation_map = {}
        
        for i, prompt_id in enumerate(prompt_ids):
            self.logger.info(f"Testing prompt {i+1}/{len(prompt_ids)} (ID: {prompt_id})...")
            
            # Load the prompt
            prompt_doc = None
            try:
                # Try to load from local file
                file_path = os.path.join("db_files", "prompts", f"{prompt_id}.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        prompt_doc = json.load(f)
            except Exception as e:
                print(f"Error loading prompt from local file: {str(e)}")
                continue
            
            if not prompt_doc:
                print(f"Could not load prompt {prompt_id}. Skipping...")
                continue
            
            # Extract the prompt text and language
            prompt_text = prompt_doc.get("prompt", "")
            language = prompt_doc.get("language", "en")
            
            if not prompt_text:
                print(f"Prompt {prompt_id} has no text. Skipping...")
                continue
            
            # Use the chatbot tester as a context manager to ensure proper resource cleanup
            try:
                with LightweightChatbotTester() as tester:
                    # Send the prompt to the chatbot
                    if language == "pt":
                        tester.test_prompt_pt(prompt_text)
                    else:
                        tester.test_prompt_en(prompt_text)
                    
                    # Store the conversation with the prompt ID
                    conversation_id = tester.store_conversation(prompt_id)
                    
                    if conversation_id:
                        conversation_map[prompt_id] = conversation_id
                        self.logger.info(f"Conversation for prompt {prompt_id} stored with ID: {conversation_id}")
                    
                    # Wait a bit to avoid overwhelming the chatbot
                    time.sleep(2)
            
            except Exception as e:
                print(f"Error testing prompt {prompt_id}: {str(e)}")
        
        return conversation_map
    
    def analyze_conversations(self, force_all: bool = False, stats_data: Dict[str, Any] = None) -> List[str]:
        """Analyze all conversation pairs for bias using both statistical and qualitative approaches.
        
        Args:
            force_all: If True, re-analyze conversations even if already analyzed
            stats_data: Optional statistical analysis data to use
            
        Returns:
            List of analysis IDs from the qualitative analysis
        """
        # Step 1: Run statistical analysis if not provided
        if not stats_data:
            stats_data = self.run_statistical_analysis()
        
        # Step 2: Find all conversation pairs using the updated BiasAnalyzer method
        self.logger.info("Finding all conversation pairs for analysis...")
        pairs = self.bias_analyzer.find_conversation_pairs(skip_analyzed=not force_all)
        if not pairs:
            self.logger.warning("No conversation pairs found for analysis.")
            return []
        
        self.logger.info(f"Found {len(pairs)} conversation pairs for analysis.")
        
        # Step 3: Analyze each pair individually
        analysis_ids = []
        for pair in pairs:
            baseline_id = pair["baseline_conversation"]["_id"]
            persona_id = pair["persona_conversation"]["_id"]
            self.logger.info(f"Analyzing conversation pair: {baseline_id} and {persona_id}")
            
            # Analyze the pair
            pair_dict = {
                "baseline_conversation": self.bias_analyzer.load_conversation(baseline_id),
                "persona_conversation": self.bias_analyzer.load_conversation(persona_id)
            }
            analysis_result = self.bias_analyzer.analyze_conversation_pair(
                pair=pair_dict,
                stats_data=stats_data
            )
            
            # Save the analysis result
            analysis_id = self.bias_analyzer.save_analysis(analysis_result)
            
            if analysis_id:
                analysis_ids.append(analysis_id)
        
        return analysis_ids
        
    def analyze_specific_conversations(self, conversation_ids: List[str], force_all: bool = False, stats_data: Dict[str, Any] = None) -> List[str]:
        """Analyze specific conversations for bias using both statistical and qualitative approaches.
        
        Args:
            conversation_ids: List of conversation IDs to analyze
            force_all: If True, re-analyze conversations even if already analyzed
            stats_data: Optional statistical analysis data to use
            
        Returns:
            List of analysis IDs from the qualitative analysis
        """
        # Step 1: Run statistical analysis on specific conversations if not provided
        if not stats_data:
            self.logger.info(f"Running statistical analysis on {len(conversation_ids)} specific conversations")
            stats_data = self.run_statistical_analysis_for_specific_conversations(conversation_ids)
        
        # Debug: Print the stats_data structure
        if stats_data:
            self.logger.info(f"Stats data contains {len(stats_data.get('results', []))} results")
            if 'results' in stats_data:
                for i, result in enumerate(stats_data['results']):
                    self.logger.info(f"Stats result {i} type: {type(result)}")
                    if isinstance(result, str):
                        self.logger.info(f"Stats result {i} is a string: {result}")
        
        # Step 2: Run qualitative analysis with statistical context
        self.logger.info(f"Analyzing {len(conversation_ids)} specific conversations for bias with statistical context")
        if force_all:
            self.logger.info("Forcing re-analysis of conversations, even those already analyzed")
            
        # Use the analyze_specific_conversation_pairs method
        return self.bias_analyzer.analyze_specific_conversation_pairs(conversation_ids, force_all=force_all, stats_data=stats_data)
    
    def run_statistical_analysis(self) -> Dict[str, Any]:
        """Run statistical analysis on all conversation pairs.
        
        Returns:
            Dictionary containing statistical analysis results
        """
        try:
            # Find conversation pairs
            self.logger.info("Finding all conversation pairs for statistical analysis...")
            conversation_pairs = []
            
            # Get conversation pairs from the bias analyzer
            if hasattr(self.bias_analyzer, 'find_conversation_pairs'):
                pairs = self.bias_analyzer.find_conversation_pairs(skip_analyzed=False)
                for pair in pairs:
                    baseline_conv = pair.get("baseline_conversation")
                    persona_conv = pair.get("persona_conversation")
                    if baseline_conv and persona_conv:
                        conversation_pairs.append((baseline_conv, persona_conv))
            
            if not conversation_pairs:
                self.logger.warning("No conversation pairs found for statistical analysis.")
                return {}
            
            self.logger.info(f"Found {len(conversation_pairs)} conversation pairs for statistical analysis.")
            
            # Run the statistical analysis
            self.logger.info("Running statistical analysis...")
            stats_results = self.statistical_analyzer.analyze_conversation_pairs(conversation_pairs)
            
            # Save the results
            stats_data = {
                "timestamp": datetime.now().isoformat(),
                "num_pairs": len(conversation_pairs),
                "results": stats_results
            }
            
            self.logger.info("Statistical analysis complete.")
            return stats_data
            
        except Exception as e:
            self.logger.error(f"Error running statistical analysis: {str(e)}")
            return {}
            
    def find_baseline_for_conversation(self, conversation_id: str) -> Optional[str]:
        """Find the baseline conversation ID for a persona conversation.
        
        Args:
            conversation_id: ID of the persona conversation
            
        Returns:
            ID of the baseline conversation, or None if not found
        """
        try:
            # Load the persona conversation
            persona_conv = None
            if hasattr(self.bias_analyzer, 'load_conversation'):
                persona_conv = self.bias_analyzer.load_conversation(conversation_id)
            
            if not persona_conv:
                self.logger.warning(f"Could not find conversation with ID: {conversation_id}")
                return None
                
            # Get the prompt ID for this conversation
            prompt_id = persona_conv.get('prompt_id')
            if not prompt_id:
                self.logger.warning(f"Conversation {conversation_id} has no prompt_id")
                return None
                
            # Load the prompt
            prompt = None
            if hasattr(self.bias_analyzer, 'load_prompt'):
                prompt = self.bias_analyzer.load_prompt(prompt_id)
            
            if not prompt:
                self.logger.warning(f"Could not find prompt with ID: {prompt_id}")
                return None
            
            # Check if the prompt has a baseline_prompt_id field (new approach)
            baseline_prompt_id = prompt.get('baseline_prompt_id')
            if baseline_prompt_id:
                self.logger.info(f"Found baseline prompt ID from persona prompt: {baseline_prompt_id}")
                # Find the baseline conversation for this prompt
                if hasattr(self.bias_analyzer, 'find_conversation_for_prompt'):
                    baseline_conv = self.bias_analyzer.find_conversation_for_prompt(baseline_prompt_id)
                    if baseline_conv:
                        baseline_conv_id = baseline_conv.get('_id')
                        return baseline_conv_id
            
            # Fallback to the old approach if baseline_prompt_id is not available
            self.logger.info("Falling back to finding baseline prompt by question matching")
            baseline_prompt = None
            if hasattr(self.bias_analyzer, 'find_baseline_prompt'):
                baseline_prompt = self.bias_analyzer.find_baseline_prompt(
                    prompt.get('question'), 
                    prompt.get('language'), 
                    prompt.get('product')
                )
            
            if not baseline_prompt:
                self.logger.warning(f"Could not find baseline prompt for question: {prompt.get('question')}")
                return None
                
            # Find the baseline conversation for this prompt
            baseline_conv_id = None
            if hasattr(self.bias_analyzer, 'find_conversation_for_prompt'):
                baseline_conv = self.bias_analyzer.find_conversation_for_prompt(baseline_prompt.get('_id'))
                if baseline_conv:
                    baseline_conv_id = baseline_conv.get('_id')
            
            if not baseline_conv_id:
                self.logger.warning(f"Could not find baseline conversation for prompt: {baseline_prompt.get('_id')}")
                return None
                
            return baseline_conv_id
            
        except Exception as e:
            self.logger.error(f"Error finding baseline for conversation {conversation_id}: {str(e)}")
            return None
    
    def run_statistical_analysis_for_specific_conversations(self, conversation_ids: List[str]) -> Dict[str, Any]:
        """Run statistical analysis on specific conversation pairs.
        
        Args:
            conversation_ids: List of conversation IDs to analyze
            
        Returns:
            Dictionary containing statistical analysis results
        """
        try:
            # Find conversation pairs for the specified conversations
            self.logger.info(f"Finding conversation pairs for {len(conversation_ids)} specific conversations...")
            conversation_pairs = []
            processed_ids = set()  # Keep track of which IDs we've already processed
            
            # First, identify which conversations are from persona prompts vs. baseline prompts
            persona_conversations = []
            baseline_conversations = []
            
            for conv_id in conversation_ids:
                # Load the conversation
                conv = None
                if hasattr(self.bias_analyzer, 'load_conversation'):
                    conv = self.bias_analyzer.load_conversation(conv_id)
                
                if not conv:
                    self.logger.warning(f"Could not find conversation with ID: {conv_id}")
                    continue
                
                # Get the prompt ID for this conversation
                prompt_id = conv.get('prompt_id')
                if not prompt_id:
                    self.logger.warning(f"Conversation {conv_id} has no prompt_id")
                    continue
                
                # Load the prompt
                prompt = None
                if hasattr(self.bias_analyzer, 'load_prompt'):
                    prompt = self.bias_analyzer.load_prompt(prompt_id)
                
                if not prompt:
                    self.logger.warning(f"Could not find prompt with ID: {prompt_id}")
                    continue
                
                # Check if this is a baseline prompt or a persona prompt
                if prompt.get('is_baseline', False):
                    baseline_conversations.append((conv_id, prompt_id))
                    self.logger.info(f"Identified conversation {conv_id} as a baseline conversation")
                else:
                    persona_conversations.append((conv_id, prompt_id, prompt.get('baseline_prompt_id')))
                    self.logger.info(f"Identified conversation {conv_id} as a persona conversation")
            
            # Now process the persona conversations to find their baseline pairs
            for persona_conv_id, persona_prompt_id, baseline_prompt_id in persona_conversations:
                if persona_conv_id in processed_ids:
                    continue  # Skip if we've already processed this ID
                
                # First try to find the baseline conversation using the baseline_prompt_id
                baseline_conv_id = None
                if baseline_prompt_id:
                    self.logger.info(f"Using baseline_prompt_id {baseline_prompt_id} to find baseline conversation")
                    # Look through our baseline_conversations list for a match
                    for b_conv_id, b_prompt_id in baseline_conversations:
                        if b_prompt_id == baseline_prompt_id:
                            baseline_conv_id = b_conv_id
                            self.logger.info(f"Found baseline conversation: {baseline_conv_id}")
                            break
                    
                    # If we didn't find it in our list, try the find_conversation_for_prompt method
                    if not baseline_conv_id and hasattr(self.bias_analyzer, 'find_conversation_for_prompt'):
                        baseline_conv = self.bias_analyzer.find_conversation_for_prompt(baseline_prompt_id)
                        if baseline_conv:
                            baseline_conv_id = baseline_conv.get('_id')
                            self.logger.info(f"Found baseline conversation via API: {baseline_conv_id}")
                
                # If we couldn't find it that way, try using find_baseline_for_conversation
                if not baseline_conv_id:
                    self.logger.info(f"Falling back to find_baseline_for_conversation for {persona_conv_id}")
                    baseline_conv_id = self.find_baseline_for_conversation(persona_conv_id)
                
                if baseline_conv_id:
                    # Get the conversations
                    baseline_conv = None
                    persona_conv = None
                    try:
                        # Load conversations directly using the bias_analyzer's load_conversation method
                        baseline_conv = self.bias_analyzer.load_conversation(baseline_conv_id)
                        persona_conv = self.bias_analyzer.load_conversation(persona_conv_id)
                        
                        if baseline_conv:
                            self.logger.info(f"Successfully loaded baseline conversation {baseline_conv_id}")
                        else:
                            self.logger.warning(f"Failed to load baseline conversation {baseline_conv_id}")
                            
                        if persona_conv:
                            self.logger.info(f"Successfully loaded persona conversation {persona_conv_id}")
                        else:
                            self.logger.warning(f"Failed to load persona conversation {persona_conv_id}")
                    except Exception as e:
                        self.logger.error(f"Error finding conversations: {str(e)}")
                        continue
                    
                    if baseline_conv and persona_conv:
                        conversation_pairs.append((baseline_conv, persona_conv))
                        processed_ids.add(persona_conv_id)
                        processed_ids.add(baseline_conv_id)
                        self.logger.info(f"Found conversation pair: baseline={baseline_conv_id}, persona={persona_conv_id}")
            
            # Also look for pairs among the baseline conversations we identified
            for baseline_conv_id, baseline_prompt_id in baseline_conversations:
                # Skip if we've already processed this ID as part of a pair
                if baseline_conv_id in processed_ids:
                    continue
                
                # Look for persona conversations that use this baseline
                for persona_conv_id, persona_prompt_id, persona_baseline_id in persona_conversations:
                    if persona_baseline_id == baseline_prompt_id and persona_conv_id not in processed_ids:
                        # Get the conversations
                        baseline_conv = None
                        persona_conv = None
                        try:
                            # Load conversations directly using the bias_analyzer's load_conversation method
                            baseline_conv = self.bias_analyzer.load_conversation(baseline_conv_id)
                            persona_conv = self.bias_analyzer.load_conversation(persona_conv_id)
                        except Exception as e:
                            self.logger.error(f"Error finding conversations: {str(e)}")
                            continue
                        
                        if baseline_conv and persona_conv:
                            conversation_pairs.append((baseline_conv, persona_conv))
                            processed_ids.add(persona_conv_id)
                            processed_ids.add(baseline_conv_id)
                            self.logger.info(f"Found conversation pair from baseline matching: baseline={baseline_conv_id}, persona={persona_conv_id}")
            
            if not conversation_pairs:
                self.logger.warning("No conversation pairs found for statistical analysis.")
                return {}
            
            self.logger.info(f"Found {len(conversation_pairs)} conversation pairs for statistical analysis.")
            
            # Run the statistical analysis
            self.logger.info("Running statistical analysis...")
            stats_results = self.statistical_analyzer.analyze_conversation_pairs(conversation_pairs)
            
            # Save the results
            stats_data = {
                "timestamp": datetime.now().isoformat(),
                "num_pairs": len(conversation_pairs),
                "results": stats_results
            }
            
            self.logger.info("Statistical analysis complete.")
            return stats_data
            
        except Exception as e:
            self.logger.error(f"Error running statistical analysis: {str(e)}")
            return {}
    
    def count_results(self):
        """Count the number of results in each category."""
        # Count personas
        personas = []
        try:
            if hasattr(self.persona_generator, 'db') and self.persona_generator.db:
                personas = self.persona_generator.db.get_all_personas()
        except Exception as e:
            self.logger.error(f"Error counting personas: {str(e)}")
        
        # Count prompts
        prompts = []
        try:
            if hasattr(self.prompt_generator, 'db') and self.prompt_generator.db:
                prompts = list(self.prompt_generator.db.prompts_collection.find({}))
        except Exception as e:
            self.logger.error(f"Error counting prompts: {str(e)}")
        
        # Count conversations
        conversations = []
        try:
            if hasattr(self.bias_analyzer, 'db') and self.bias_analyzer.db:
                conversations = list(self.bias_analyzer.db.conversations_collection.find({}))
        except Exception as e:
            self.logger.error(f"Error counting conversations: {str(e)}")
        
        # Count analyses
        analyses = []
        try:
            if hasattr(self.bias_analyzer, 'db') and self.bias_analyzer.db:
                analyses = list(self.bias_analyzer.db.test_results_collection.find({"analysis_type": "bias_analysis"}))
        except Exception as e:
            self.logger.error(f"Error counting analyses: {str(e)}")
        
        self.logger.info("=== System Status ===")
        self.logger.info(f"Personas: {len(personas)}")
        self.logger.info(f"Prompts: {len(prompts)}")
        self.logger.info(f"Conversations: {len(conversations)}")
        self.logger.info(f"Bias analyses: {len(analyses)}")

def main():
    """Run the bias tester."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run bias testing for Aurora chatbot")
    parser.add_argument("--products", type=int, default=2, help="Number of products to test")
    parser.add_argument("--personas", type=int, default=3, help="Number of personas to generate")
    parser.add_argument("--questions", type=int, default=3, help="Number of questions per product")
    parser.add_argument("--diversity", type=str, default="balanced", 
                        choices=["conservative", "balanced", "creative"],
                        help="Diversity strategy for persona generation")
    parser.add_argument("--temperature", type=float, help="Specific temperature to use for generation")
    parser.add_argument("--enforce-diversity", action="store_true", help="Enforce diversity between generated personas")
    parser.add_argument("--no-diversity", dest="enforce_diversity", action="store_false", help="Don't enforce diversity between generated personas")
    parser.add_argument("--force-all", action="store_true", help="Process all existing data (not just new data) in each step and force re-analysis of all conversations")
    parser.add_argument("--clean", action="store_true", help="Clean all JSON files before running")
    
    # Add skip options
    parser.add_argument("--skip-personas", action="store_true", help="Skip persona generation")
    parser.add_argument("--skip-prompts", action="store_true", help="Skip prompt generation")
    parser.add_argument("--skip-testing", action="store_true", help="Skip chatbot testing")
    parser.add_argument("--skip-stats", action="store_true", help="Skip statistical analysis")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip qualitative bias analysis")
    
    # Add workflow options
    persona_group = parser.add_mutually_exclusive_group()
    persona_group.add_argument("--test-existing", action="store_true", help="Test existing personas only (don't generate new ones)")
    persona_group.add_argument("--test-new-only", action="store_true", help="Test only newly generated personas (default behavior)")
    persona_group.add_argument("--test-all", action="store_true", help="Test all personas (existing and new)")
    
    # Force options
    parser.add_argument("--force-analysis", action="store_true", help="Force re-analysis of conversations even if already analyzed")
    
    parser.set_defaults(enforce_diversity=True)
    
    args = parser.parse_args()
    
    # Clean JSON files if requested
    if args.clean:
        clean_json_files()
    
    # Set up logging
    logger = get_run_logger()
    
    # Generate a unique run ID
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting bias test run {run_id} with JSON database")
    
    # Initialize the bias tester with the logger
    tester = BiasTester(logger=logger)
    
    if args.skip_prompts and args.skip_testing and args.skip_analysis:
        logger.error("You've chosen to skip all steps. Nothing to do.")
        sys.exit(1)
    
    # Track newly generated personas for this run
    new_persona_ids = []
    
    # Step 1: Generate personas based on workflow mode
    personas = tester.persona_generator.load_all_personas()
    logger.info(f"Loaded {len(personas)} existing personas")
    
    if not args.skip_personas:
        if args.test_existing:
            # Don't generate any new personas, only use existing ones
            logger.info("Using existing personas only (--test-existing mode)")
            new_persona_ids = []
        elif args.test_all:
            # Generate new personas and also use existing ones
            logger.info(f"Generating {args.personas} new personas and will also use existing ones (--test-all mode)")
            new_persona_ids = tester.generate_personas(
                args.personas,
                diversity_strategy=args.diversity,
                temperature=args.temperature
            )
            logger.info(f"Generated {len(new_persona_ids)} new personas: {new_persona_ids}")
        else:  # Default or --test-new-only
            # Generate and only use new personas
            logger.info(f"Generating {args.personas} new personas...")
            new_persona_ids = tester.generate_personas(
                args.personas,
                diversity_strategy=args.diversity,
                temperature=args.temperature
            )
            logger.info(f"Generated {len(new_persona_ids)} new personas: {new_persona_ids}")
    else:
        logger.info(f"Already have {len(personas)} personas, which meets the requested {args.personas}. No new personas generated.")
        logger.info(f"Use --test-new-only to force generating new personas, or --test-existing to test existing ones.")
    
    # Determine which personas to use for this run
    personas_for_this_run = []
    if args.test_all or args.force_all:
        # Use all available personas
        personas_for_this_run = [p.get('_id') for p in tester.persona_generator.load_all_personas()]
        logger.info(f"Using all {len(personas_for_this_run)} available personas (--test-all mode)")
    elif args.test_existing:
        # Use all existing personas (but not any new ones)
        existing_persona_ids = [p.get('_id') for p in personas]
        personas_for_this_run = existing_persona_ids
        logger.info(f"Using {len(personas_for_this_run)} existing personas (--test-existing mode)")
    else:
        # Default: use only newly generated personas
        personas_for_this_run = new_persona_ids
        logger.info(f"Using only {len(personas_for_this_run)} newly generated personas (--test-new-only mode)")
    
    # Track prompts generated for this run
    new_prompt_ids = []
    
    if not args.skip_prompts:
        # Step 2: Generate prompts only for the selected personas
        if personas_for_this_run:
            logger.info(f"Generating prompts for {len(personas_for_this_run)} personas and {args.products} products")
            new_prompt_ids = tester.prompt_generator.generate_prompts_for_specific_personas(
                personas_for_this_run,
                args.products,
                questions_per_product=args.questions
            )
    
    if not args.skip_testing:
        # If we skipped prompt generation, load all prompts
        if args.skip_prompts:
            prompt_ids = []
            try:
                if hasattr(tester.prompt_generator.db, 'prompts_collection'):
                    cursor = tester.prompt_generator.db.prompts_collection.find({})
                    for doc in cursor:
                        if '_id' in doc:
                            prompt_ids.append(str(doc['_id']))
            except Exception as e:
                print(f"Error loading prompts from database: {str(e)}")
        
        # Step 3: Test prompts with the chatbot
        if args.force_all:
            # Test all prompts if force_all is set
            all_prompts = tester.prompt_generator.load_all_prompts()
            all_prompt_ids = [p.get('_id') for p in all_prompts]
            logger.info(f"Testing all {len(all_prompt_ids)} available prompts due to --force-all flag")
            tester.new_conversation_ids = tester.test_prompts(all_prompt_ids)
        else:
            # Test only the new prompts by default, but make sure to include baseline prompts
            # for each persona prompt to ensure we have complete conversation pairs
            baseline_prompt_ids = []
            
            # Find the corresponding baseline prompts for each persona prompt
            if new_prompt_ids:
                all_prompts = tester.prompt_generator.load_all_prompts()
                for prompt in all_prompts:
                    if prompt.get('_id') in new_prompt_ids:
                        # This is one of our new persona prompts, get its baseline_prompt_id
                        if not prompt.get('is_baseline', False):
                            # Use the baseline_prompt_id field directly if available
                            baseline_prompt_id = prompt.get('baseline_prompt_id')
                            if baseline_prompt_id:
                                logger.info(f"Found baseline prompt ID {baseline_prompt_id} for persona prompt {prompt.get('_id')}")
                                baseline_prompt_ids.append(baseline_prompt_id)
                            else:
                                # Fallback to the old approach if baseline_prompt_id is not available
                                logger.info(f"No baseline_prompt_id found for prompt {prompt.get('_id')}, using matching approach")
                                for baseline_prompt in all_prompts:
                                    if (baseline_prompt.get('is_baseline', False) and
                                        baseline_prompt.get('question') == prompt.get('question') and
                                        baseline_prompt.get('language') == prompt.get('language') and
                                        baseline_prompt.get('product') == prompt.get('product')):
                                        baseline_prompt_ids.append(baseline_prompt.get('_id'))
                                        break
            
            # Combine persona prompts and their baseline prompts
            prompts_to_test = list(set(new_prompt_ids + baseline_prompt_ids))
            logger.info(f"Testing {len(new_prompt_ids)} newly generated prompts plus {len(baseline_prompt_ids)} corresponding baseline prompts")
            tester.new_conversation_ids = tester.test_prompts(prompts_to_test)
    
    # Run statistical analysis
    stats_data = {}
    if args.skip_stats:
        print("Skipping statistical analysis")
    elif args.force_all:
        # Run statistical analysis on all conversations
        logger.info("Running statistical analysis on all conversations due to --force-all flag")
        stats_data = tester.run_statistical_analysis()
    elif tester.new_conversation_ids:
        # Run statistical analysis on only the new conversations
        logger.info(f"Running statistical analysis on {len(tester.new_conversation_ids)} new conversations")
        stats_data = tester.run_statistical_analysis_for_specific_conversations(
            list(tester.new_conversation_ids.values())
        )
    else:
        logger.warning("No new conversations for statistical analysis. Use --force-all to analyze existing conversations.")
        # Debug: Print the stats_data structure
        if stats_data:
            logger.info(f"Stats data contains {len(stats_data.get('results', []))} results")
            if 'results' in stats_data:
                for i, result in enumerate(stats_data['results']):
                    logger.info(f"Stats result {i} type: {type(result)}")
                    if isinstance(result, str):
                        logger.info(f"Stats result {i} is a string: {result}")
    
    # Run bias analysis
    analysis_ids = []
    if args.skip_analysis:
        print("Skipping qualitative bias analysis")
    elif args.force_all:
        # Analyze all conversation pairs
        logger.info("Analyzing all conversation pairs due to --force-all flag")
        analysis_ids = tester.analyze_conversations(force_all=args.force_analysis, stats_data=stats_data)
    elif tester.new_conversation_ids:
        # Analyze only the new conversations
        logger.info(f"Analyzing {len(tester.new_conversation_ids)} new conversations")
        analysis_ids = tester.analyze_specific_conversations(
            list(tester.new_conversation_ids.values()),
            force_all=args.force_analysis,
            stats_data=stats_data
        )
    else:
        logger.warning("No new conversations to analyze. Use --force-all to analyze existing conversations.")
    # Log completion of the analysis
    if analysis_ids:
        logger.info(f"Completed {len(analysis_ids)} qualitative bias analyses.")
    if stats_data:
        logger.info("Statistical analysis completed successfully.")
        
    # Log completion of the run
    logger.info(f"Bias test run {run_id} completed successfully")

def clean_json_files():
    """Clean all JSON files in the db_files directory."""
    import shutil
    
    db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_files")
    if os.path.exists(db_dir):
        print(f"Cleaning JSON files in {db_dir}...")
        for subdir in ["prompts", "personas", "convos", "test_results", "stats", "results"]:
            subdir_path = os.path.join(db_dir, subdir)
            if os.path.exists(subdir_path):
                print(f"Removing files in {subdir_path}...")
                for filename in os.listdir(subdir_path):
                    if filename.endswith(".json"):
                        file_path = os.path.join(subdir_path, filename)
                        os.remove(file_path)
                        print(f"Removed {file_path}")
    
    print("JSON files cleaned successfully.")

if __name__ == "__main__":
    main()
