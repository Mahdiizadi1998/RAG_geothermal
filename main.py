"""Main entry point for RAG Geothermal Wells system."""

import argparse
import yaml
from pathlib import Path
from agents.ingestion_agent import IngestionAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.extraction_agent import ParameterExtractionAgent


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_pdf(pdf_path: str, config: dict) -> dict:
    """
    Process a PDF document through the full pipeline.
    
    Args:
        pdf_path: Path to PDF file
        config: Configuration dictionary
        
    Returns:
        Dictionary with extracted parameters
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Step 1: Ingestion
    print("  [1/3] Ingesting PDF...")
    ingestion_agent = IngestionAgent()
    doc_data = ingestion_agent.process_document(pdf_path)
    print(f"       Extracted {len(doc_data['pages'])} pages")
    
    # Step 2: Preprocessing
    print("  [2/3] Preprocessing and chunking...")
    preprocessing_agent = PreprocessingAgent(config.get('extraction', {}).get('chunking', {}))
    chunks = preprocessing_agent.process_document(doc_data['pages'], strategy='table-aware')
    print(f"       Created {len(chunks)} chunks")
    
    # Step 3: Parameter Extraction
    print("  [3/3] Extracting parameters...")
    extraction_agent = ParameterExtractionAgent(config.get('extraction', {}))
    params = extraction_agent.extract_from_chunks(chunks)
    
    # Print results
    print("\n=== Extraction Results ===")
    print(f"Well Name: {doc_data['metadata'].well_name or 'Not detected'}")
    print(f"Trajectory Points: {len(params.trajectory)}")
    print(f"Casing Sections: {len(params.casing)}")
    print(f"Reservoir Pressure: {params.reservoir_pressure} bar" if params.reservoir_pressure else "Pressure: Not found")
    print(f"Reservoir Temperature: {params.reservoir_temperature}°C" if params.reservoir_temperature else "Temperature: Not found")
    
    # Validation results
    if params.validation_results:
        print("\n=== Validation ===")
        for i, result in enumerate(params.validation_results):
            print(f"Validation {i+1}: {'PASSED' if result.is_valid else 'FAILED'} (confidence: {result.confidence:.2f})")
            if result.errors:
                for error in result.errors:
                    print(f"  ERROR: {error}")
            if result.warnings:
                for warning in result.warnings:
                    print(f"  WARNING: {warning}")
    
    return {
        'metadata': doc_data['metadata'],
        'parameters': params,
        'chunks': chunks
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='RAG system for extracting geothermal well parameters from PDFs'
    )
    parser.add_argument(
        'pdf_path',
        type=str,
        help='Path to PDF file to process'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for extracted parameters (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Check if PDF exists
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return 1
    
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Warning: Config file not found: {args.config}")
        print("Using default configuration")
        config = {}
    
    # Process PDF
    try:
        results = process_pdf(str(pdf_path), config)
        
        # Save output if requested
        if args.output:
            import json
            from dataclasses import asdict
            
            output_data = {
                'metadata': {
                    'filename': results['metadata'].filename,
                    'num_pages': results['metadata'].num_pages,
                    'well_name': results['metadata'].well_name
                },
                'parameters': {
                    'trajectory': [
                        {'md': p.md, 'tvd': p.tvd, 'inclination': p.inclination}
                        for p in results['parameters'].trajectory
                    ],
                    'casing': [
                        {'top_depth': c.top_depth, 'bottom_depth': c.bottom_depth, 'diameter': c.diameter}
                        for c in results['parameters'].casing
                    ],
                    'reservoir_pressure': results['parameters'].reservoir_pressure,
                    'reservoir_temperature': results['parameters'].reservoir_temperature
                }
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\nResults saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
