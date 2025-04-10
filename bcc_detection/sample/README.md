# BCC Detection Sample Project

This sample project demonstrates the complete pipeline for Basal Cell Carcinoma (BCC) detection using Whole Slide Images (WSIs). The project includes all stages from preprocessing to evaluation.

## Project Structure

```
sample/
├── data/           # Sample data directory
├── outputs/        # Output files and results
├── docs/           # Documentation and reports
├── sample_pipeline.py      # Main pipeline script
├── sample_data_loader.py   # Data loading utilities
└── requirements.txt        # Project dependencies
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the complete pipeline:
```bash
python sample_pipeline.py
```

This will:
- Create sample data if not present
- Run through all 5 stages of the pipeline
- Generate visualizations and documentation
- Save results in the outputs directory

## Pipeline Stages

1. **Preprocessing**
   - Tissue segmentation
   - Patch extraction
   - Quality filtering

2. **Feature Extraction**
   - Deep feature extraction using EfficientNet-B7
   - Fuzzy clustering for pattern analysis

3. **Model Training**
   - Training with mixed precision
   - Early stopping
   - Model checkpointing

4. **Evaluation**
   - Performance metrics
   - Uncertainty estimation
   - Visualization of results

5. **Documentation**
   - Generate comprehensive report
   - Save visualizations
   - Document metrics and results

## Outputs

The pipeline generates:
- Preprocessing visualizations
- Feature extraction results
- Training curves
- Evaluation metrics
- Documentation in Markdown format

## Sample Data

The project includes a sample dataset generator that creates:
- Sample WSI images
- Patches with labels
- Train/validation/test splits

## Notes

- This is a sample implementation for demonstration purposes
- For production use, replace the sample data with real WSI data
- Adjust parameters in the configuration as needed
- The pipeline is designed to be modular and extensible 