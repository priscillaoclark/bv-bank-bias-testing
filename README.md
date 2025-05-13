# Aurora Chatbot Bias Testing System

A comprehensive framework for detecting and analyzing potential biases in the Aurora chatbot responses across diverse personas, products, and languages.

## Overview

The Aurora Chatbot Bias Testing System is designed to systematically test and analyze potential biases in AI-powered chatbot responses. It generates diverse personas, creates targeted prompts, tests the chatbot with these prompts, and performs both statistical and qualitative analysis of the responses to identify patterns of bias.

This system helps ensure that the Aurora chatbot provides fair, unbiased, and consistent responses to all users, regardless of their demographic characteristics. By simulating interactions from diverse user personas, the system can detect subtle biases in tone, information depth, personalization, and other aspects of the chatbot's responses.

## Key Features

- **Diverse Persona Generation**: Creates realistic personas with varying demographics using Google Gemini API. Personas include attributes such as age, gender, education level, income, location, occupation, digital literacy, and financial knowledge.

- **Flexible Prompt Creation**: Generates both baseline and persona-specific prompts for multiple banking products. Each persona-specific prompt is linked to a baseline prompt for fair comparison. Supports both English and Portuguese languages.

- **Automated Chatbot Testing**: Tests the Aurora chatbot with generated prompts and records responses using Selenium-based browser automation. Handles authentication, session management, and conversation tracking.

- **Statistical Analysis**: Performs quantitative analysis of response differences using multiple metrics:
  - Sentiment analysis using NLTK's VADER algorithm
  - Response metrics (length, word count, sentence count)
  - Readability analysis using multiple standardized formulas
  - Word frequency analysis to detect terminology differences
  - Similarity analysis using cosine similarity with TF-IDF

- **Qualitative Bias Detection**: Uses Google Gemini API to analyze conversation pairs for potential bias across multiple criteria:
  - Tone and engagement
  - Personalization
  - Depth of information
  - Inclusivity/neutrality
  - Response consistency
  - Disparate impact analysis

- **Comprehensive Reporting**: Generates detailed reports of bias analysis results, including:
  - Summary statistics across all analyses
  - Statistical metrics with interpretations
  - Detailed analysis by persona, product, and language
  - Criteria-based analysis with ratings and explanations

- **Flexible Workflow Options**: Supports multiple testing scenarios:
  - Default workflow: Process only newly generated data
  - Test existing workflow: Process only existing personas
  - Test new only workflow: Process only newly generated personas
  - Comprehensive workflow: Process all available data

- **Simple Storage**: Uses local JSON files for all data storage without external database dependencies. All data is organized in the `db_files` directory with clear subdirectories for each data type.

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│ Persona         │────▶│ Prompt          │────▶│ Chatbot         │
│ Generator       │     │ Generator       │     │ Tester          │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│ Report          │◀────│ Bias           │◀────│ Statistical      │
│ Generator       │     │ Analyzer        │     │ Bias Analyzer   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              ▲
                              │
                      ┌───────┴───────┐
                      │               │
                      │  Local JSON   │
                      │  Storage      │
                      │               │
                      └───────────────┘
```

## Getting Started

### Prerequisites

- **Python 3.8+**
- **Required Python packages**:
  - google-generativeai (for persona generation and bias analysis)
  - selenium (for chatbot testing)
  - nltk (for natural language processing)
  - scikit-learn (for TF-IDF vectorization and similarity analysis)
  - pandas (for data manipulation)
  - numpy (for numerical operations)
  - textstat (for readability metrics)
  - python-dotenv (for environment variable management)
- **Chrome browser and ChromeDriver** (for Selenium-based chatbot testing)
- **Google Gemini API key** (for persona generation and qualitative analysis)
- **Aurora chatbot credentials** (for testing the chatbot)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/bv-bank-2.git
cd bv-bank-2
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file with the following variables:

```
GEMINI_API_KEY=your_gemini_api_key
AURORA_USERNAME=your_aurora_username
AURORA_PASSWORD=your_aurora_password
AURORA_BV_URL=https://aurora.jabuti.ai/Aurora_BV_2025
```

4. **Create directory structure**

The system will automatically create the necessary directory structure on first run, but you can also create it manually:

```bash
mkdir -p db_files/personas db_files/prompts db_files/convos db_files/results db_files/stats reports
```

### Basic Usage

1. **Run a complete bias test with default settings**:

```bash
python run_bias_test.py
```

This will generate 2 personas, create prompts for 3 products, test them with the Aurora chatbot, and analyze the results.

2. **Generate more diverse personas**:

```bash
python run_bias_test.py --personas 5 --diversity balanced
```

3. **Test specific products with existing personas**:

```bash
python run_bias_test.py --products 2 --test-existing
```

4. **Generate a comprehensive report from analysis results**:

```bash
python generate_report.py
```

5. **Clean all data and start fresh**:

```bash
python clean_data.py
```

For more detailed usage instructions, see the [Run Bias Test Guide](docs/run_bias_test_guide.md).

## Core Components

### Persona Generator (`persona_generator.py`)

Generates diverse personas with varying demographics, including:
- **Demographics**: Age, gender, location
- **Socioeconomic factors**: Education level, income level, occupation
- **Digital capabilities**: Digital literacy level
- **Financial context**: Financial knowledge, banking habits

Key features:
- Uses Google Gemini API with different temperature settings to control diversity
- Validates personas to ensure sufficient diversity in the test set
- Supports different diversity strategies: mixed, balanced, diverse, homogeneous
- Stores personas as JSON files with unique IDs

### Prompt Generator (`prompt_generator.py`)

Creates prompts for testing the chatbot:
- **Baseline prompts**: Generic questions without persona context
- **Persona-specific prompts**: Questions framed in the context of a specific persona
- Supports multiple banking products from BV Bank's product lineup
- Generates prompts in both English and Portuguese

Key features:
- Links each persona-specific prompt to a baseline prompt for fair comparison
- Uses `baseline_prompt_id` field for reliable conversation pair matching
- Can generate prompts for all personas or only specific personas
- Creates multiple questions per product to test different aspects

### Chatbot Tester (`chatbot_tester.py`)

Tests the Aurora chatbot with generated prompts:
- Uses Selenium WebDriver for browser automation
- Sends prompts to the chatbot in both English and Portuguese
- Records complete conversation history including all turns
- Captures metadata about the conversation context

Key features:
- Implements context manager pattern for proper resource cleanup
- Handles authentication and session management automatically
- Stores conversations with links to their corresponding prompts
- Uses headless browser mode for efficient testing

### Statistical Bias Analyzer (`statistical_bias_analyzer.py`)

Performs quantitative analysis of chatbot responses:
- **Sentiment analysis**: Uses NLTK's VADER algorithm to detect emotional bias
- **Response metrics**: Analyzes length, word count, and sentence count differences
- **Readability analysis**: Calculates multiple readability scores (Flesch-Kincaid, Gunning Fog, etc.)
- **Word frequency analysis**: Identifies terminology differences after removing stopwords
- **Similarity analysis**: Uses cosine similarity with TF-IDF to measure content differences

Key features:
- Calculates differences between baseline and persona-specific responses
- Provides interpretation guidelines for each metric
- Aggregates results by demographic groups for pattern detection
- Stores detailed analysis results for each conversation pair

### Bias Analyzer (`bias_analyzer.py`)

Conducts qualitative analysis of potential bias:
- Uses Google Gemini API to analyze conversation pairs
- Evaluates responses across six key criteria:
  - **Tone and engagement**: Emotional quality and style of response
  - **Personalization**: Adaptation to user-specific details
  - **Depth of information**: Completeness and clarity of information
  - **Inclusivity/neutrality**: Avoidance of stereotypes and assumptions
  - **Response consistency**: Consistency across different user profiles
  - **Disparate impact analysis**: Potential differential impacts on users

Key features:
- Incorporates statistical context into the qualitative analysis
- Provides ratings (1-5) and detailed explanations for each criterion
- Calculates overall bias scores for each conversation pair
- Identifies specific examples of potential bias in responses

### Data Storage (`storage/json_database.py`)

The system uses local JSON files to store all data, organized in the `db_files` directory:

- `db_files/personas`: Stores generated personas with demographic information
- `db_files/prompts`: Stores baseline and persona-specific prompts with product and language metadata
- `db_files/convos`: Stores conversation records from chatbot testing, including all turns and metadata
- `db_files/results`: Stores qualitative bias analysis results with ratings and explanations
- `db_files/stats`: Stores statistical analysis results with metrics and differences

Key features:
- Simple file-based storage approach eliminates external database dependencies
- Clear organization with separate directories for each data type
- Consistent JSON format for all data with unique IDs for referencing
- Support for both direct file access and programmatic API

### Report Generator (`generate_report.py`)

Creates comprehensive reports of bias analysis results:
- **Summary statistics**: Overall bias scores and patterns across demographic groups
- **Statistical metrics**: Quantitative measures of bias with interpretations
- **Detailed analysis**: In-depth analysis by persona, product, and language
- **Criteria-based analysis**: Breakdown of bias by evaluation criteria

Key features:
- Loads and aggregates results from all analysis files
- Organizes findings by persona characteristics for pattern identification
- Includes both statistical and qualitative insights in a single report
- Saves reports to the `reports` directory with timestamps for tracking

## Documentation

The system includes comprehensive documentation to help you understand and use all features:

- [Run Bias Test Guide](docs/run_bias_test_guide.md): Detailed instructions for using the system, including command-line arguments, workflow options, and troubleshooting tips.
- [Statistical Methods](docs/statistical_methods.md): In-depth documentation of all statistical methods used for bias detection, including metrics, interpretation guidelines, and implementation details.
- [System Architecture](docs/system_architecture.md): Comprehensive overview of the system architecture, component interactions, data flow, and integration points.

## BV Bank Products

The system is configured to test the following BV Bank products:

1. Digital bank account
2. Credit cards
3. Personal loans
4. Payroll loans
5. Auto equity loans
6. Vehicle financing
7. Solar panel financing
8. Private party auto financing
9. Auto insurance
10. Solar panel insurance
11. Credit card protection insurance
12. Financial protection insurance
13. Personal accident insurance
14. Dental insurance
15. Assistance services
16. Investment products

## Workflow Examples

### Example 1: Initial Testing with New Personas

```bash
# Generate 5 personas with balanced diversity and test 3 products
python run_bias_test.py --personas 5 --products 3 --diversity balanced

# Generate a report of the results
python generate_report.py
```

### Example 2: Testing New Products with Existing Personas

```bash
# Test 2 specific products with all existing personas
python run_bias_test.py --products 2 --test-existing

# Generate a report of the results
python generate_report.py
```

### Example 3: Comprehensive Testing of All Data

```bash
# Test all personas (existing and new) with all products
python run_bias_test.py --force-all

# Generate a report of the results
python generate_report.py
```

## Troubleshooting

### Common Issues

1. **Persona Generation Failures**
   - Check your Google Gemini API key in the `.env` file
   - Verify your API quota and limits
   - Try reducing the number of personas or using a lower temperature

2. **Chatbot Testing Errors**
   - Verify your Aurora credentials in the `.env` file
   - Check that Chrome and ChromeDriver are properly installed
   - Ensure you have a stable internet connection

3. **Missing Conversation Pairs**
   - Check the `baseline_prompt_id` field in persona prompts
   - Verify that both baseline and persona-specific prompts were tested
   - Run with `--force-all` to ensure all prompts are tested

4. **Analysis Failures**
   - Ensure NLTK resources are properly downloaded
   - Check your Google Gemini API key and quota
   - Verify that conversations were properly recorded

### Getting Help

If you encounter issues not covered here, please check the detailed documentation in the `docs` folder or contact the system maintainers.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API for persona generation and bias analysis
- Selenium WebDriver for chatbot testing automation
- NLTK and scikit-learn for statistical analysis
- textstat for readability metrics
- All contributors who have helped improve this system