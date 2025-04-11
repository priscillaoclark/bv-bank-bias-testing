#!/usr/bin/env python
"""
Test script for the enhanced persona generator functionality.
This script generates a single persona and displays the result.
"""

import json
from persona_generator import PersonaGenerator

def main():
    """Run a test of the persona generator."""
    print("Initializing PersonaGenerator...")
    generator = PersonaGenerator()
    
    print("\nLoading existing personas for comparison...")
    existing_personas = generator.load_personas()
    print(f"Found {len(existing_personas)} existing personas")
    
    print("\nGenerating a new persona with diversity validation...")
    persona = generator.ensure_diverse_persona(
        existing_personas,
        diversity_level="balanced",
        max_attempts=3
    )
    
    print("\nGenerated Persona:")
    # Format the persona for display, excluding large metadata fields
    display_persona = {k: v for k, v in persona.items() if k not in ["raw_response"]}
    print(json.dumps(display_persona, indent=2))
    
    print("\nSaving persona to database and local file...")
    persona_id = generator.save_persona(persona)
    print(f"Persona saved with ID: {persona_id}")
    
    return persona_id

if __name__ == "__main__":
    main()
