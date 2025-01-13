# Academic Paper Analysis Pipeline

A machine learning pipeline that analyzes academic papers to determine their publishability and recommend suitable conferences for submission.

## Features

- **Publishability Classification**: Uses BERT embeddings to assess if a paper is ready for publication
- **Conference Recommendation**: Combines vector similarity and Gemini AI analysis to suggest appropriate conferences
- **Flexible Input**: Process papers from local directories or Google Drive
- **Performance Metrics**: Provides detailed cross-validation metrics and processing statistics
- **Structured Output**: Generates CSV reports with analysis results

## Setup

1. Clone the repository:
```bash
git clone https://github.com/googleboy-byte/research-grading-ai-pipeline
cd <research-grading-ai-pipeline>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment:
- Create `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_key_here
```
- For Google Drive integration, place your `credentials.json` in the root directory

4. Prepare data directories:
```
data/
├── Train/
│   ├── Publishable/
│   └── Non-Publishable/
└── Test/
```

## Usage

### Basic Usage
```bash
python run.py --input-dir data/Test --threshold 0.7
```

### Google Drive Integration
```bash
python run.py --gdrive-folder your_folder_id --threshold 0.7
```

### Arguments
- `--input-dir`: Local directory containing PDFs (default: data/Test)
- `--gdrive-folder`: Google Drive folder ID for input PDFs
- `--threshold`: Publishability classification threshold (default: 0.7)

## Output

The script generates:
1. CSV file with results (`combined_analysis_results_TIMESTAMP.csv`)
2. Detailed log file (`outputs/run_output_TIMESTAMP.txt`)

### Output Columns
- `pdfname`: Name of the processed PDF
- `publishable`: Binary classification result (0/1)
- `conference`: Recommended conference (if publishable)
- `justification`: Explanation for the recommendation
- `publishability_score`: Confidence score (0-1)
- `processing_time`: Time taken to process the paper

## Model Training

The repository includes a pre-trained publishability classifier. To retrain:
1. Place training papers in appropriate directories under `data/Train/`
2. Run the training script (not included in basic distribution)
