# End-to-End Testing Guide for Aurora Chatbot Bias Testing

This guide provides step-by-step instructions for testing the entire bias testing process from persona generation to bias analysis with a single persona and question.

## Prerequisites

Before starting, ensure you have:

1. MongoDB running locally or a connection to a remote MongoDB instance
2. Python 3.8+ installed
3. Required Python packages installed (see `requirements.txt`)
4. API keys configured in your `.env` file:
   - `GEMINI_API_KEY` for persona generation and bias analysis
   - Any keys required for the Aurora chatbot testing

## Step 1: Generate a Single Persona

The first step is to generate a diverse persona that will be used to test the Aurora chatbot.

```bash
# Run the test_persona_generator.py script
python test_persona_generator.py
```

This script will:
1. Initialize the PersonaGenerator
2. Load any existing personas for comparison
3. Generate a new, diverse persona
4. Save the persona to both MongoDB and a local file
5. Return the ID of the generated persona

**Example output:**
```
Initializing PersonaGenerator...
MongoDB connection available. Will save personas to both MongoDB and local files.

Loading existing personas for comparison...
Loaded 2 personas from MongoDB
Found 2 existing personas

Generating a new persona with diversity validation...
Generating persona with temperature 0.40 from user-specified (0.4)
Generated diverse persona on attempt 1: Similarity score: 0.10. Persona is sufficiently different from existing personas.

Generated Persona:
{
  "name": "Carlos Eduardo Mendes",
  "age": 42,
  "gender": "Male",
  "occupation": "Small business owner (Auto repair shop)",
  "education_level": "Technical school (automotive mechanics)",
  ...
}

Saving persona to database and local file...
Saved persona to MongoDB with ID: 7f8a3b2d-9c5e-4d6a-8f7g-1h2j3k4l5m6n
Persona saved with ID: 7f8a3b2d-9c5e-4d6a-8f7g-1h2j3k4l5m6n
```

Make note of the persona ID as you'll need it in the next steps.

## Step 2: Generate a Test Prompt

Next, generate a test prompt for the persona using the PromptGenerator.

```bash
# Run the test_prompt_generator.py script with the persona ID
python -c "from prompt_generator import PromptGenerator; pg = PromptGenerator(); prompt_id = pg.generate_prompt_for_persona('7f8a3b2d-9c5e-4d6a-8f7g-1h2j3k4l5m6n', 'auto_insurance', 'en'); print(f'Generated prompt ID: {prompt_id}')"
```

Replace `7f8a3b2d-9c5e-4d6a-8f7g-1h2j3k4l5m6n` with your actual persona ID, and you can choose a different product from BV Bank's offerings:

- `digital_bank_account`
- `credit_cards`
- `personal_loans`
- `payroll_loans`
- `auto_equity_loans`
- `vehicle_financing`
- `solar_panel_financing`
- `private_party_auto_financing`
- `auto_insurance`
- `solar_panel_insurance`
- `credit_card_protection_insurance`
- `financial_protection_insurance`
- `personal_accident_insurance`
- `dental_insurance`
- `assistance_services`
- `investment_products`

Also, choose the language: `en` for English or `pt` for Portuguese.

**Example output:**
```
Generated prompt ID: 9d8c7b6a-5f4e-3d2c-1b0a-9z8y7x6w5v4u
```

Make note of the prompt ID for the next step.

## Step 3: Generate a Baseline Prompt

For comparison, you also need to generate a baseline prompt (non-persona specific) for the same product.

```bash
# Generate a baseline prompt for the same product
python -c "from prompt_generator import PromptGenerator; pg = PromptGenerator(); baseline_id = pg.generate_baseline_prompt('auto_insurance', 'en'); print(f'Generated baseline prompt ID: {baseline_id}')"
```

**Example output:**
```
Generated baseline prompt ID: 5a4b3c2d-1e0f-9g8h-7i6j-5k4l3m2n1o0p
```

Make note of the baseline prompt ID for the next steps.

## Step 4: Test the Prompts with the Aurora Chatbot

Now, test both prompts with the Aurora chatbot to get responses.

```bash
# Test the persona-specific prompt
python -c "from chatbot_tester import LightweightChatbotTester; with LightweightChatbotTester() as tester: tester.test_prompt_by_id('9d8c7b6a-5f4e-3d2c-1b0a-9z8y7x6w5v4u'); persona_conv_id = tester.store_conversation('9d8c7b6a-5f4e-3d2c-1b0a-9z8y7x6w5v4u'); print(f'Persona conversation ID: {persona_conv_id}')"

# Test the baseline prompt
python -c "from chatbot_tester import LightweightChatbotTester; with LightweightChatbotTester() as tester: tester.test_prompt_by_id('5a4b3c2d-1e0f-9g8h-7i6j-5k4l3m2n1o0p'); baseline_conv_id = tester.store_conversation('5a4b3c2d-1e0f-9g8h-7i6j-5k4l3m2n1o0p'); print(f'Baseline conversation ID: {baseline_conv_id}')"
```

Replace the prompt IDs with your actual prompt IDs from steps 2 and 3.

**Example output:**
```
Persona conversation ID: 1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p
Baseline conversation ID: 6p5o4n3m-2l1k-0j9i-8h7g-6f5e4d3c2b1a
```

Make note of both conversation IDs for the final step.

## Step 5: Analyze the Conversations for Bias

Finally, analyze the pair of conversations for potential bias using both statistical and qualitative approaches.

```bash
# Run the statistical analysis first
python -c "from statistical_bias_analyzer import StatisticalBiasAnalyzer; analyzer = StatisticalBiasAnalyzer(); stats_data = analyzer.analyze_conversation_pair(('6p5o4n3m-2l1k-0j9i-8h7g-6f5e4d3c2b1a', '1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p')); print('Statistical analysis complete')"

# Then run the qualitative bias analysis
python -c "from bias_analyzer import BiasAnalyzer; analyzer = BiasAnalyzer(); analysis_id = analyzer.analyze_specific_conversation_pair('6p5o4n3m-2l1k-0j9i-8h7g-6f5e4d3c2b1a', '1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p', force=True); print(f'Bias analysis complete. Analysis ID: {analysis_id}')"
```

Replace the conversation IDs with your actual conversation IDs from step 4.

**Example output:**
```
Statistical analysis complete
Bias analysis complete. Analysis ID: 0z9y8x7w-6v5u-4t3s-2r1q-0p9o8n7m6l5k
```

## Step 6: View the Analysis Results

You can view the analysis results in MongoDB or in the local JSON files:

```bash
# View the analysis results from the local file
cat db_files/results/bias_analysis_0z9y8x7w-6v5u-4t3s-2r1q-0p9o8n7m6l5k.json
```

Replace the analysis ID with your actual analysis ID from step 5.

## Running the Entire Process with a Single Command

For convenience, you can also run the entire process with a single command using the `run_bias_test.py` script:

```bash
python run_bias_test.py --personas 1 --products 1 --diversity balanced --force-all
```

This will:
1. Generate 1 persona with balanced diversity
2. Generate prompts for 1 product
3. Test the prompts with the Aurora chatbot
4. Run statistical and qualitative bias analysis
5. Save all results to MongoDB and local files

## Troubleshooting

If you encounter any issues during the testing process:

1. **MongoDB Connection Issues**:
   - Check that MongoDB is running
   - Verify your connection string in the `.env` file
   - Ensure the database name is correct (`rai_testing2`)

2. **API Key Issues**:
   - Verify that your Gemini API key is valid
   - Check that all required API keys are set in the `.env` file

3. **Missing Data**:
   - If analysis results are missing, check that all previous steps completed successfully
   - Verify that the IDs you're using are correct
   - Check the MongoDB collections and local files for the expected data

4. **Error Messages**:
   - For detailed error information, check the console output
   - Look for specific error messages that indicate what went wrong

## Next Steps

After completing a successful end-to-end test, you can:

1. Generate more personas with different attributes
2. Test with different products
3. Compare bias analysis results across different demographic groups
4. Identify patterns of bias that may need attention

For batch testing with multiple personas and products, use the `run_bias_test.py` script with appropriate parameters.
