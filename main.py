# main.py

import argparse
import logging
import os
import sys

# Ensure 'src' is in the Python path for imports if running from project root.
# This is often handled by IDEs, but can be explicit for clarity or direct script runs.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)  # Adds project root

try:
    from src.config_management.config_loader import load_main_config
    from src.utils.logger_setup import setup_logging
    from src.orchestrator import execute_framework_mode  # Import the function from orchestrator
except ImportError as e:
    # This basic print is for when logging itself might not be set up yet.
    print(f"Critical Import Error: {e}. Ensure 'src' is discoverable or PYTHONPATH is set.")
    print(f"If running from project root, 'python main.py' should work.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# Logger for this main module. It will be configured by setup_logging().
logger = logging.getLogger(__name__)  # Gets a logger named 'main' or '__main__'


def main():
    """Main entry point for the A²IR framework."""
    # Setup logging as the first step.
    # This will use 'config/logging_config.yaml' if found, otherwise defaults.
    setup_logging()

    parser = argparse.ArgumentParser(
        description="A²IR Framework: Neurosymbolic and Agentic AI for Cybersecurity Incident Response."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="config/main_config.yaml",
        help="Path to the main YAML configuration file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="preprocess_data",  # A safe default to start with
        choices=["preprocess_data", "train_alert_classifier", "build_knowledge_graph",
                 "run_incident_response", "evaluate_framework"],
        help="The operational mode for the A²IR framework."
    )

    # Mode-specific arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to process in 'preprocess_data' mode."
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default=None,
        help="Path to training data for 'train_alert_classifier' mode."
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data for 'evaluate_framework' mode."
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to ground truth data for 'evaluate_framework' mode."
    )
    parser.add_argument(
        "--incident-id",
        type=str,
        default=None,
        help="Specific incident ID/file to process in 'run_incident_response' mode."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory or file path."
    )
    parser.add_argument(
        "--threat-intel",
        type=str,
        default=None,
        help="Additional threat intelligence file for 'build_knowledge_graph' mode."
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate visualization plots in 'evaluate_framework' mode."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging."
    )

    args = parser.parse_args()

    # Adjust logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Now that logging is set up, this log message will use the configured settings.
    logger.info(f"A²IR Framework invoked with arguments: {args}")
    logger.info(f"Current working directory: {os.getcwd()}")

    try:
        # Load configuration
        config = load_main_config(args.config_file)
        if not config:
            logger.error(f"Failed to load configuration from {args.config_file}. Exiting.")
            sys.exit(1)

        # Merge command-line arguments with config if needed
        if args.generate_plots:
            if 'evaluation' not in config:
                config['evaluation'] = {}
            config['evaluation']['generate_plots'] = True

        # Call the function from the orchestrator module
        # This function contains the logic for different modes
        execute_framework_mode(args, config)

    except KeyboardInterrupt:
        logger.warning("A²IR Framework execution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        # Catch any unhandled exceptions from the orchestrator or config loading
        logger.critical("A critical error occurred in the A²IR framework's main execution.", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("A²IR Framework execution cycle complete.")


if __name__ == "__main__":
    main()