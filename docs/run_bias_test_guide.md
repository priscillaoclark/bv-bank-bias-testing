# Run Bias Test Guide

This guide provides detailed instructions for using the `run_bias_test.py` script to perform end-to-end testing of the Aurora chatbot bias testing system.

## Overview

The `run_bias_test.py` script automates the entire bias testing process, including:

1. Generating diverse personas
2. Creating prompts for various banking products
3. Testing the prompts with the Aurora chatbot
4. Analyzing the responses for bias using both statistical and qualitative methods

This script is ideal for batch testing with multiple personas and products to identify potential bias patterns in the chatbot's responses.

## Prerequisites

Before running the script, ensure you have:

1. MongoDB running locally or a connection to a remote MongoDB instance
2. Python 3.8+ installed
3. Required Python packages installed (see `requirements.txt`)
4. API keys configured in your `.env` file:
   - `GEMINI_API_KEY` for persona generation and bias analysis
   - Any keys required for the Aurora chatbot testing

## Basic Usage

The simplest way to run the script is:

```bash
python run_bias_test.py
```

This will use default values for all parameters:
- 3 personas with mixed diversity
- 3 products
- No forced re-analysis of existing results

## Command-Line Arguments

The script supports several command-line arguments to customize the testing process:

```bash
python run_bias_test.py [--personas NUM_PERSONAS] [--products NUM_PRODUCTS] [--diversity DIVERSITY_STRATEGY] [--force-all] [--temperature TEMPERATURE] [--language LANGUAGE]
```

### Required Arguments

None. All arguments have default values.

### Optional Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--personas` | Number of personas to generate | 3 | Any positive integer |
| `--products` | Number of products to test | 3 | Any positive integer |
| `--diversity` | Diversity strategy for persona generation | "mixed" | "mixed", "conservative", "balanced", "creative" |
| `--force-all` | Force re-analysis of all conversations | False | Flag (no value needed) |
| `--temperature` | Specific temperature for persona generation | None | Float between 0.0 and 1.0 |
| `--language` | Language for prompts and testing | "en" | "en" (English), "pt" (Portuguese) |
| `--log-level` | Logging level | "INFO" | "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" |
| `--run-id` | Custom run ID for logging | Auto-generated | Any string identifier |

## Examples

### Generate 5 Personas with Balanced Diversity for 2 Products

```bash
python run_bias_test.py --personas 5 --products 2 --diversity balanced
```

This command will:
- Generate 5 personas with balanced diversity (medium temperature range)
- Test 2 randomly selected products from BV Bank's offerings
- Use English as the default language
- Skip analysis for conversation pairs that have already been analyzed

### Test All Products with Creative Personas in Portuguese

```bash
python run_bias_test.py --personas 3 --products 16 --diversity creative --language pt
```

This command will:
- Generate 3 personas with creative diversity (high temperature range)
- Test all 16 products in BV Bank's offerings
- Use Portuguese for prompts and testing
- Skip analysis for conversation pairs that have already been analyzed

### Force Re-Analysis of All Conversations

```bash
python run_bias_test.py --personas 2 --products 2 --force-all
```

This command will:
- Generate 2 personas with mixed diversity
- Test 2 randomly selected products
- Force re-analysis of all conversation pairs, even if they've been analyzed before

### Use a Specific Temperature for Persona Generation

```bash
python run_bias_test.py --personas 3 --temperature 0.8
```

This command will:
- Generate 3 personas using a fixed temperature of 0.8 (more creative/diverse)
- Test 3 randomly selected products
- Skip analysis for conversation pairs that have already been analyzed

## Process Flow

When you run the script, it follows this process:

1. **Persona Generation**:
   - Generates the specified number of personas using the PersonaGenerator
   - Applies diversity validation to ensure personas are sufficiently different
   - Saves personas to MongoDB and local files

2. **Prompt Generation**:
   - For each persona, generates prompts for the specified number of products
   - Also generates baseline (non-persona specific) prompts for the same products
   - Saves prompts to MongoDB and local files

3. **Chatbot Testing**:
   - Sends each prompt to the Aurora chatbot
   - Captures and stores the conversation
   - Saves conversations to MongoDB and local files

4. **Statistical Analysis**:
   - Analyzes conversation pairs using statistical methods
   - Measures sentiment differences, response length variations, etc.
   - Aggregates metrics by demographic groups

5. **Qualitative Analysis**:
   - Uses Gemini to analyze conversation pairs for bias
   - Evaluates based on criteria like tone, personalization, depth of information
   - Incorporates statistical context into the analysis

6. **Results Storage**:
   - Saves all analysis results to MongoDB and local files
   - Generates unique IDs for all artifacts

## Output Files

The script generates several output files in the `db_files` directory:

- `db_files/personas/`: JSON files for each generated persona
- `db_files/prompts/`: JSON files for each generated prompt
- `db_files/convos/`: JSON files for each conversation with the chatbot
- `db_files/stats/`: JSON files with statistical analysis results
- `db_files/results/`: JSON files with qualitative bias analysis results

Additionally, the script generates log files in the `logs` directory:

- `logs/bias_test_run_[RUN_ID].log`: Detailed log file for each run

## Monitoring Progress

The script provides detailed logging during execution, both to the console and to log files:

- Progress indicators for each phase
- IDs of generated artifacts
- Error messages if any step fails
- Summary of results at the end

### Logging

Each run of the script generates a unique log file in the `logs` directory. The log file is named using the run ID, which is either auto-generated or specified via the `--run-id` parameter.

#### Log Levels

You can control the verbosity of logging using the `--log-level` parameter:

- `DEBUG`: Detailed debugging information
- `INFO`: General information about the progress (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical error messages

#### Example with Custom Run ID and Debug Logging

```bash
python run_bias_test.py --personas 2 --products 2 --run-id "test_run_001" --log-level DEBUG
```

This will create a log file at `logs/bias_test_run_test_run_001.log` with detailed debug information.

## Troubleshooting

### Common Issues

1. **MongoDB Connection Errors**:
   ```
   Error connecting to MongoDB: connection refused
   ```
   **Solution**: Ensure MongoDB is running and the connection string in your `.env` file is correct.

2. **API Key Errors**:
   ```
   Error: GEMINI_API_KEY environment variable is not set.
   ```
   **Solution**: Add your Gemini API key to the `.env` file.

3. **Timeout Errors**:
   ```
   Error testing prompt: Timeout waiting for response
   ```
   **Solution**: The Aurora chatbot may be overloaded. Try reducing the number of products or adding delays between requests.

4. **Memory Issues**:
   ```
   MemoryError: ...
   ```
   **Solution**: Reduce the number of personas and products, or run the script on a machine with more memory.

### Debugging Tips

1. **Enable Verbose Logging**:
   - Add `--verbose` flag to see more detailed logs (if implemented)
   - Check the console output for specific error messages

2. **Check MongoDB Collections**:
   - Use MongoDB Compass to inspect the collections
   - Verify that documents are being created correctly

3. **Examine Local Files**:
   - Check the JSON files in the `db_files` directory
   - Look for any malformed or incomplete files

4. **Run Individual Components**:
   - Test each component separately (persona generation, prompt generation, etc.)
   - Identify which specific step is failing

## Advanced Usage

### Custom Product Selection

By default, the script randomly selects products from BV Bank's offerings. To test specific products, you can modify the script to use a predefined list:

```python
# In run_bias_test.py
specific_products = ["auto_insurance", "personal_loans", "credit_cards"]
tester.generate_prompts(specific_products)
```

### Custom Persona Attributes

To focus on specific demographic attributes or bias factors, you can modify the `PERSONA_ATTRIBUTES` dictionary in `persona_generator.py`.

### Integration with CI/CD

For automated testing in a CI/CD pipeline, you can run the script with specific parameters and logging options:

```bash
# In a CI/CD script
python run_bias_test.py --personas 2 --products 2 --diversity balanced --force-all --run-id "ci_run_${BUILD_ID}" --log-level INFO
```

This approach allows you to:
- Use a consistent naming convention for log files
- Correlate log files with specific CI/CD builds
- Adjust logging verbosity based on the environment

## Conclusion

The `run_bias_test.py` script provides a comprehensive solution for end-to-end testing of the Aurora chatbot for potential bias. By adjusting the parameters, you can test various scenarios and identify patterns of bias across different demographic groups and banking products.

For more detailed information about the individual components of the bias testing system, refer to the documentation for each module:

- `persona_generator.py`: Generates diverse personas for testing
- `prompt_generator.py`: Creates prompts for various banking products
- `chatbot_tester.py`: Tests the Aurora chatbot with generated prompts
- `statistical_bias_analyzer.py`: Performs statistical analysis of chatbot responses
- `bias_analyzer.py`: Conducts qualitative bias analysis using Gemini
