# src/orchestrator.py

import logging
import os
import sys
import json
from typing import Any, Dict, List, Optional  # Added List, Optional
from datetime import datetime

# Component imports
from src.data_processing.data_loader import run_all_preprocessing  # This now handles all datasets
# We might not need to import run_alert_preprocessing directly here anymore if run_all_preprocessing is the main entry.
# from src.data_processing.alert_preprocessor import run_preprocessing as run_alert_preprocessing # Potentially remove
from src.data_processing.data_loader import _load_and_preprocess_single_dataset  # NEW: For specific dataset processing
from src.data_processing.alert_preprocessor import \
    AlertPreprocessor  # NEW: To pass to _load_and_preprocess_single_dataset

from src.knowledge_graph.ftags.ftag_initializer import initialize_knowledge_graph
from src.neural_components.alert_classification.trainer import train as train_classifier
from src.agents.coordination.incident_workflow import \
    run_workflow  # IncidentWorkflow class might not be directly needed here
from src.evaluation.framework_evaluator import run_evaluation

logger = logging.getLogger(__name__)


def execute_framework_mode(args: Any, config: Dict[str, Any]):
    """
    Executes the A²IR framework based on the specified mode and configuration.

    This is the main orchestrator that coordinates all components based on
    the selected operational mode.

    Args:
        args: Parsed command-line arguments (should have a 'mode' attribute)
        config: The loaded main configuration
    """
    logger.info(f"Orchestrator executing mode: {args.mode}")

    # Initialize directories and check dependencies once at the start if appropriate
    # For now, assuming these are handled elsewhere or implicitly.
    # initialize_directories(config)
    # validate_dependencies()

    try:
        if args.mode == "preprocess_data":
            execute_preprocessing(args, config)

        elif args.mode == "train_alert_classifier":
            execute_training(args, config)

        elif args.mode == "build_knowledge_graph":
            execute_knowledge_graph_build(args, config)

        elif args.mode == "run_incident_response":
            execute_incident_response(args, config)

        elif args.mode == "evaluate_framework":
            execute_evaluation(args, config)

        else:
            logger.error(f"Unknown or not-yet-implemented mode: {args.mode}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error in mode '{args.mode}': {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Orchestrator finished mode '{args.mode}' successfully")


def execute_preprocessing(args: Any, config: Dict[str, Any]):
    """Execute data preprocessing mode."""
    logger.info("Starting data preprocessing...")

    datasets_config = config.get('datasets', {})  # Moved up

    if not datasets_config:
        logger.error("No datasets configured in main_config.yaml")
        return

    # Initialize the general alert standardizer once, as it might be needed by specialized preprocessors
    general_alert_standardizer = AlertPreprocessor(config.get('preprocessing', {}))

    # Check if we should preprocess all datasets or specific ones
    if hasattr(args, 'dataset') and args.dataset:
        # Process specific dataset
        dataset_name = args.dataset
        if dataset_name not in datasets_config:
            logger.error(f"Dataset '{dataset_name}' not found in configuration")
            return

        logger.info(f"Preprocessing specific dataset: {dataset_name}")
        # Use the _load_and_preprocess_single_dataset from data_loader.py
        # This function now expects the alert_standardizer
        processed_alerts = _load_and_preprocess_single_dataset(
            dataset_name,
            config,
            general_alert_standardizer  # Pass the standardizer
        )

        if processed_alerts:
            logger.info(f"Preprocessed {len(processed_alerts)} alerts from {dataset_name}")
            # Determine output path for specific dataset processing
            # The saving logic is now inside _load_and_preprocess_single_dataset,
            # but if you want to override it via command line:
            output_path_arg = args.output if hasattr(args, 'output') and args.output else None
            if output_path_arg:
                logger.info(f"Attempting to save specifically to: {output_path_arg}")
                os.makedirs(os.path.dirname(output_path_arg), exist_ok=True)
                try:
                    with open(output_path_arg, 'w') as f:
                        json.dump(processed_alerts, f, indent=2)
                    logger.info(f"Orchestrator: Saved specific dataset processed alerts to {output_path_arg}")
                except Exception as e:
                    logger.error(f"Orchestrator: Could not save specific dataset processed alerts: {e}")
            # Else, it's saved to the default location within _load_and_preprocess_single_dataset
        else:
            logger.warning(f"No alerts processed for dataset: {dataset_name}")

    else:
        # Process all configured datasets
        logger.info("Preprocessing all configured datasets...")
        # run_all_preprocessing now expects the standardizer to be passed down
        # or it initializes it itself. Let's assume the latter for now based on its current structure.
        # If run_all_preprocessing was changed to take standardizer, pass it:
        # all_processed_data = run_all_preprocessing(config, general_alert_standardizer)
        all_processed_data = run_all_preprocessing(config)  # Assuming run_all_preprocessing handles standardizer init

        if all_processed_data:
            for name, data_list in all_processed_data.items():
                logger.info(f"Orchestrator: Completed preprocessing for {name}, found {len(data_list)} alerts.")
        else:
            logger.info("Orchestrator: run_all_preprocessing completed, no data returned or processed.")


def execute_training(args: Any, config: Dict[str, Any]):
    """Execute alert classifier training mode."""
    logger.info("Starting alert classifier training...")

    model_config = config.get('models', {}).get('alert_classifier', {})
    training_config = config.get('training', {})  # This might contain paths or params

    # Determine training data path:
    # Priority: command line arg -> config file -> default
    default_training_data_path = "data/processed/training/all_processed_alerts.json"  # Example default

    # Check if a specific dataset was processed and use its output if no explicit training data path is given
    # This logic requires knowing which dataset was just preprocessed if args.dataset was used in a prior step.
    # For simplicity, we'll rely on args.training_data or a configured path.

    training_data_path = default_training_data_path  # Fallback default
    if hasattr(args, 'training_data') and args.training_data:
        training_data_path = args.training_data
    elif training_config.get('training_data_path'):  # New config option
        training_data_path = training_config.get('training_data_path')
    elif hasattr(args, 'dataset') and args.dataset:  # If a single dataset was processed
        # Construct path based on the dataset processed
        # This assumes a consistent output naming from the preprocessing step
        specific_dataset_processed_path = f"data/processed/{args.dataset}/{args.dataset}_processed_alerts.json"
        if os.path.exists(specific_dataset_processed_path):
            training_data_path = specific_dataset_processed_path
            logger.info(f"Using recently preprocessed data for training: {training_data_path}")
        else:
            logger.warning(f"Processed data for {args.dataset} not found at {specific_dataset_processed_path}. "
                           f"Falling back to default/configured training data path.")

    if not os.path.exists(training_data_path):
        logger.error(f"Training data not found at: {training_data_path}")
        logger.info("Please run preprocessing first or specify a valid training data path "
                    "via --training-data or in the 'training' section of main_config.yaml.")
        return

    logger.info(f"Using training data from: {training_data_path}")

    # Merge configurations for the trainer
    # The train_classifier function from trainer.py now expects a dict with 'model', 'training', 'output_dir'
    trainer_config = {
        'model': model_config,
        'training': training_config,  # Pass general training params like epochs, batch_size
        'output_dir': model_config.get('path', 'models/alert_classifier')
    }

    # Train the classifier
    classifier = train_classifier(trainer_config, training_data_path)  # Pass the config dict

    if classifier:
        logger.info("Alert classifier training completed.")
    else:
        logger.error("Alert classifier training failed.")


def execute_knowledge_graph_build(args: Any, config: Dict[str, Any]):
    """Execute knowledge graph building mode."""
    logger.info("Starting knowledge graph construction...")

    kg_config = config.get('knowledge_graph', {})
    ftag = initialize_knowledge_graph(config)  # This function already takes the main config

    if not ftag:
        logger.error("Failed to initialize knowledge graph.")
        return

    save_path = kg_config.get('storage_path', 'models/knowledge_graph/a2ir_kg.json')
    # Ensure directory exists before trying to save
    save_dir = os.path.dirname(save_path)
    if save_dir:  # Check if save_dir is not empty (e.g. if save_path is just a filename)
        os.makedirs(save_dir, exist_ok=True)
    else:  # Handle case where save_path might be just a filename in the current directory
        save_dir = "."

    if ftag.save_to_file(save_path):
        logger.info(f"Knowledge graph saved to: {save_path}")
    else:
        logger.error(f"Failed to save knowledge graph to {save_path}")
        return  # Exit if KG saving fails, as stats depend on it

    if hasattr(args, 'threat_intel') and args.threat_intel:
        logger.info(f"Loading additional threat intelligence from: {args.threat_intel}")
        logger.warning("Additional threat intelligence loading not yet implemented in orchestrator.")
        # if ftag.load_additional_intel(args.threat_intel): # Assuming such a method exists in FTAG
        #    ftag.save_to_file(save_path) # Re-save if updated
        #    logger.info(f"Knowledge graph updated with {args.threat_intel} and re-saved.")

    # Generate statistics about the knowledge graph
    if ftag.graph:  # Check if graph object exists
        stats = {
            'nodes': ftag.graph.number_of_nodes(),
            'edges': ftag.graph.number_of_edges(),
            'timestamp': datetime.now().isoformat()
        }
        stats_file = os.path.join(save_dir, 'kg_statistics.json')  # Use determined save_dir
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Knowledge graph statistics: {stats} saved to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save KG statistics: {e}")
    else:
        logger.warning("Knowledge graph object does not have a 'graph' attribute or is None. Skipping stats.")


def execute_incident_response(args: Any, config: Dict[str, Any]):
    """Execute incident response simulation mode."""
    logger.info("Starting incident response simulation...")

    # Determine input source
    incident_file_path = None
    if hasattr(args, 'incident_id') and args.incident_id:
        incident_file_path = args.incident_id
        logger.info(f"Processing specific incident/file: {incident_file_path}")
    else:
        # Use default from config or a fallback
        incident_file_path = config.get('incident_response', {}).get('test_incidents_path',
                                                                     'data/test/sample_incidents.json')
        logger.info(f"Using default incident file: {incident_file_path}")

    if not os.path.exists(incident_file_path):
        logger.error(f"Incident file not found: {incident_file_path}")
        return

    # Set up output directory
    default_output_dir = os.path.join('data', 'results', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir = args.output if hasattr(args, 'output') and args.output else default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    # Run the workflow (run_workflow from incident_workflow.py)
    # The 'process' mode in run_workflow needs to align with how it uses kwargs
    run_workflow(config, 'process', input_file=incident_file_path, output_dir=output_dir)

    logger.info(f"Incident response simulation completed.")  # Message about where results are saved is in run_workflow


def execute_evaluation(args: Any, config: Dict[str, Any]):
    """Execute framework evaluation mode."""
    logger.info("Starting framework evaluation...")

    eval_config = config.get('evaluation', {})
    test_data_path = args.test_data if hasattr(args, 'test_data') and args.test_data else \
        eval_config.get('test_data_path', 'data/test/evaluation_data.json')  # Changed key for consistency
    ground_truth_path = args.ground_truth if hasattr(args, 'ground_truth') and args.ground_truth else \
        eval_config.get('ground_truth_path', 'data/test/ground_truth.json')  # Changed key

    if not os.path.exists(test_data_path):
        logger.error(f"Test data not found: {test_data_path}")
        return

    if ground_truth_path and not os.path.exists(
            ground_truth_path):  # Check if ground_truth_path is not None before os.path.exists
        logger.warning(f"Ground truth file specified but not found: {ground_truth_path}. Proceeding without it.")
        ground_truth_path = None
    elif not ground_truth_path:
        logger.info(
            "No ground truth path specified. Evaluation will proceed without ground truth labels for some metrics.")

    # Run evaluation
    # run_evaluation function from framework_evaluator.py
    evaluation_results = run_evaluation(config, test_data_path, ground_truth_path)

    if evaluation_results:
        logger.info("Framework evaluation completed.")
        # Saving summary is handled within run_evaluation or its sub-components
        # If you want to explicitly log where the summary is:
        summary_path = evaluation_results.get('summary_report_path', 'data/evaluation/evaluation_summary.md')  # Example
        if os.path.exists(summary_path):
            logger.info(f"Evaluation summary report available at: {summary_path}")
    else:
        logger.error("Framework evaluation failed or produced no results.")


def check_system_readiness(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Check if the system is ready for operation by verifying component availability.
    Args:
        config: Application configuration
    Returns:
        Dictionary indicating readiness of each component
    """
    readiness = {
        'config_valid': False,  # Start assuming not valid
        'classifier_model': False,
        'knowledge_graph': False,
        'asp_rules': False,
    }

    # Basic config validation
    required_sections = ['datasets', 'models', 'knowledge_graph', 'agents',
                         'reasoning']  # Added 'reasoning' for ASP rules
    if all(section in config for section in required_sections):
        readiness['config_valid'] = True
    else:
        missing = [section for section in required_sections if section not in config]
        logger.warning(f"Configuration is missing required sections: {missing}")
        # Even if config is not fully valid for all operations, some checks can proceed.

    # Check if classifier model exists
    # Ensure 'models' and 'alert_classifier' keys exist before accessing 'path'
    model_config = config.get('models', {}).get('alert_classifier', {})
    model_dir_path = model_config.get('path', 'models/alert_classifier')  # Directory where model parts are stored
    model_file_path = os.path.join(model_dir_path, 'alert_classifier.pkl')  # Assuming this is the main model file
    if os.path.exists(model_file_path):
        readiness['classifier_model'] = True
    else:
        logger.warning(f"Classifier model not found at {model_file_path}")

    # Check if knowledge graph exists
    kg_config = config.get('knowledge_graph', {})
    kg_path = kg_config.get('storage_path', 'models/knowledge_graph/a2ir_kg.json')
    if os.path.exists(kg_path):
        readiness['knowledge_graph'] = True
    else:
        logger.warning(f"Knowledge graph not found at {kg_path}")

    # Check if ASP rules exist
    reasoning_config = config.get('reasoning', {})
    # rules_path in config could be a specific file or a directory. ASPReasoner._load_rule_sets expects a base path.
    # Let's assume rules_path points to a directory containing .lp files or a primary .lp file.
    # The current ASPReasoner._load_rule_sets logic checks for specific filenames in the dir of self.rules_path
    rules_base_path = reasoning_config.get('rules_path', 'data/knowledge_base/asp_rules.lp')
    rules_dir = os.path.dirname(rules_base_path)  # Get directory
    if os.path.isdir(rules_dir) and any(f.endswith('.lp') for f in os.listdir(rules_dir)):
        readiness['asp_rules'] = True
    elif os.path.exists(rules_base_path) and rules_base_path.endswith('.lp'):  # If it's a single file
        readiness['asp_rules'] = True
    else:
        logger.warning(
            f"ASP rules directory ({rules_dir}) or main rule file ({rules_base_path}) not found or empty of .lp files.")

    return readiness


def initialize_directories(config: Dict[str, Any]):
    """Create necessary directories for the framework."""
    # Base directories from config or defaults
    data_base = config.get('paths', {}).get('data_dir', 'data')
    models_base = config.get('paths', {}).get('models_dir', 'models')
    logs_base = config.get('paths', {}).get('logs_dir', 'logs')

    directories = [
        os.path.join(data_base, 'raw'),
        os.path.join(data_base, 'processed', 'training'),  # Specific path for training data output
        os.path.join(data_base, 'test'),
        os.path.join(data_base, 'results'),
        os.path.join(data_base, 'evaluation', 'plots'),
        os.path.join(data_base, 'evaluation', 'reports'),
        os.path.join(data_base, 'knowledge_base'),  # For ASP rules etc.
        os.path.join(models_base, 'alert_classifier'),
        os.path.join(models_base, 'knowledge_graph'),
        logs_base
    ]

    # Add dataset specific processed directories
    for dataset_name in config.get('datasets', {}).keys():
        directories.append(os.path.join(data_base, 'processed', dataset_name))

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
        except OSError as e:
            logger.error(f"Could not create directory {directory}: {e}")
            # Depending on severity, you might want to exit or continue


def validate_dependencies():
    """Check if required dependencies are installed."""
    required_modules = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'networkx': 'networkx',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'yaml': 'pyyaml',
        'requests': 'requests',  # Added for neural_investigator
        'dnspython': 'dnspython',  # For dns.resolver
        'python-whois': 'python-whois'  # For whois
        # clingo is a command-line tool, not a python lib to import directly usually
    }

    missing_modules = []

    for module, package in required_modules.items():
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(package)

    # Check for clingo executable
    try:
        import subprocess
        subprocess.run(['clingo', '--version'], capture_output=True, check=True)
        logger.info("Clingo executable found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Clingo executable not found or not working. Please ensure it's installed and in your PATH.")
        # This is a critical dependency for symbolic reasoning.
        # Depending on strictness, you might sys.exit(1) here.

    if missing_modules:
        logger.error(f"Missing required Python dependencies: {', '.join(missing_modules)}")
        logger.info(f"Please install them, e.g., with: pip install {' '.join(missing_modules)}")
        # sys.exit(1) # Optional: exit if critical dependencies are missing


# This main guard is for running orchestrator.py directly for its utility modes
if __name__ == "__main__":
    # This setup is usually done when main.py calls the orchestrator.
    # If running orchestrator.py directly, we need to set up paths and logging.
    SCRIPT_DIR_ORCH = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_ORCH = os.path.dirname(SCRIPT_DIR_ORCH)  # Assuming src is one level down from project root
    if PROJECT_ROOT_ORCH not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_ORCH)

    from src.utils.logger_setup import setup_logging  # Import after path setup
    from src.config_management.config_loader import load_main_config  # Import after path setup

    setup_logging()  # Use default logging config path

    # Parse arguments for orchestrator's own main function
    import argparse

    parser = argparse.ArgumentParser(description="A²IR Framework Orchestrator Utilities")
    parser.add_argument(
        '--utility-mode',  # Changed argument name to avoid conflict with framework modes
        choices=['check_readiness', 'initialize_dirs', 'validate_deps', 'list_modes'],  # Added validate_deps
        help='Orchestrator utility mode'
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="config/main_config.yaml",  # Default config for utilities
        help="Path to the main YAML configuration file for utility operations."
    )

    utility_args = parser.parse_args()

    # Load configuration using the potentially overridden config file
    # Ensure load_main_config can handle the path correctly if it's not project root relative
    # For now, assume config_file is relative to where orchestrator.py is run, or absolute
    config_to_use = load_main_config(utility_args.config_file)
    if not config_to_use:
        logger.error(f"Failed to load configuration from {utility_args.config_file} for utility mode. Exiting.")
        sys.exit(1)

    if utility_args.utility_mode == 'check_readiness':
        readiness = check_system_readiness(config_to_use)
        print("\nSystem Readiness Check:")
        for component, ready in readiness.items():
            status = "✓" if ready else "✗"
            print(f"  {component}: {status}")

        if all(readiness.values()):
            print("\nSystem is fully ready for operation!")
        else:
            print("\nSystem is not fully ready. Please run initialization or check missing components.")

    elif utility_args.utility_mode == 'initialize_dirs':
        print("Initializing A²IR framework directories...")
        initialize_directories(config_to_use)
        print("Directory initialization complete!")

    elif utility_args.utility_mode == 'validate_deps':
        print("Validating dependencies...")
        validate_dependencies()
        print("Dependency validation complete. Check logs for details.")


    elif utility_args.utility_mode == 'list_modes':
        print("\nAvailable A²IR Framework Operational Modes (run via main.py --mode <mode_name>):")
        print("  - preprocess_data: Preprocess security alert data")
        print("  - train_alert_classifier: Train the neural alert classifier")
        print("  - build_knowledge_graph: Build the FTAG knowledge graph")
        print("  - run_incident_response: Run incident response simulation")
        print("  - evaluate_framework: Evaluate the complete framework")

    elif utility_args.utility_mode is None:
        parser.print_help()