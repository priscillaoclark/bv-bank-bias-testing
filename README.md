# Aurora Chatbot Bias Testing System

A comprehensive framework for detecting and analyzing potential biases in the Aurora chatbot responses across diverse personas, products, and languages.

## Overview

The Aurora Chatbot Bias Testing System is designed to systematically test and analyze potential biases in AI-powered chatbot responses. It generates diverse personas, creates targeted prompts, tests the chatbot with these prompts, and performs both statistical and qualitative analysis of the responses to identify patterns of bias.

## Key Features

- **Diverse Persona Generation**: Creates realistic personas with varying demographics using Google Gemini API
- **Flexible Prompt Creation**: Generates both baseline and persona-specific prompts for multiple banking products
- **Automated Chatbot Testing**: Tests the Aurora chatbot with generated prompts and records responses
- **Statistical Analysis**: Performs quantitative analysis of response differences using multiple metrics
- **Qualitative Bias Detection**: Uses AI to analyze conversation pairs for potential bias
- **Comprehensive Reporting**: Generates detailed reports of bias analysis results
- **Flexible Workflow**: Supports both focused (new data) and comprehensive (all data) testing scenarios
- **Simple Storage**: Uses local JSON files for all data storage without external database dependencies

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

- Python 3.8+
- Google Gemini API key
- Access to Aurora chatbot

### Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/bv-bank-2.git
cd bv-bank-2
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set up environment variables

Create a `.env` file with the following variables:

```
GEMINI_API_KEY=your_gemini_api_key
```

### Basic Usage

Run a complete bias test with default settings:

```bash
python run_bias_test.py
```

Generate a report from the analysis results:

```bash
python generate_report.py
```

For more detailed usage instructions, see the [Run Bias Test Guide](docs/run_bias_test_guide.md).

## Core Components

### Persona Generator

Generates diverse personas with varying demographics, including:
- Age, gender, location
- Education level, income level
- Occupation, digital literacy
- Financial knowledge, banking habits

### Prompt Generator

Creates prompts for testing the chatbot:
- Baseline prompts (without persona context)
- Persona-specific prompts for each persona
- Supports multiple banking products and languages
- Links persona prompts to baseline prompts for analysis

### Chatbot Tester

Tests the Aurora chatbot with generated prompts:
- Sends prompts to the chatbot
- Records responses and conversation details
- Handles authentication and session management

### Statistical Bias Analyzer

Performs quantitative analysis of chatbot responses:
- Sentiment analysis (using VADER)
- Response metrics (length, word count, sentence count)
- Readability analysis (Flesch-Kincaid, Gunning Fog, etc.)
- Word frequency analysis
- Similarity analysis (cosine similarity with TF-IDF)

### Bias Analyzer

Conducts qualitative analysis of potential bias:
- Compares baseline and persona-specific responses
- Evaluates based on criteria like tone, personalization, depth of information
- Incorporates statistical context into the analysis
- Provides detailed explanations and examples

### Data Storage

The system uses local JSON files to store all data, organized in the `db_files` directory:

- `db_files/personas`: Stores generated personas
- `db_files/prompts`: Stores baseline and persona-specific prompts
- `db_files/convos`: Stores conversation records from chatbot testing
- `db_files/results`: Stores bias analysis results
- `db_files/stats`: Stores statistical analysis results
- `db_files/results/prompts`: Stores generated prompts for bias analysis

This simple file-based storage approach eliminates the need for external database dependencies while maintaining all functionality.

### Report Generator

Creates comprehensive reports of bias analysis results:
- Summary statistics across all analyses
- Detailed analysis by persona, product, and language
- Statistical metrics with interpretations
- Qualitative analysis with ratings and explanations

## Documentation

- [Run Bias Test Guide](docs/run_bias_test_guide.md): Detailed instructions for using the system
- [Statistical Methods](docs/statistical_methods.md): Documentation of statistical methods used for bias detection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API for persona generation and bias analysis
- MongoDB for data storage
- NLTK and scikit-learn for statistical analysis
- textstat for readability metrics