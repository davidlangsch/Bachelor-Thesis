import pm4py
import pandas as pd
import time
import os
import csv

from pm4py.conformance import fitness_alignments, precision_alignments, fitness_token_based_replay, \
    precision_token_based_replay, generalization_tbr
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.ilp import algorithm as ilp_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer

def load_event_log(file_path):
    """
    Load an event log from a XES file.

    :param file_path: Path to the event log

    :return event_log: PM4Py event log
    """
    event_log = pm4py.read_xes(file_path)
    return event_log


def discover_models(event_log, output_dir):
    """
    Discover process models (petri nets) using Alpha Miner, Inductive Miner, Heuristics Miner and ILP Miner
    and measure their execution time.

    :param event_log: PM4Py event log
    :param output_dir: Path to the output directory

    :return dictionary: Dictionary containing nets, initial and final markings
    """

    execution_times = {}

    # Alpha Miner
    alpha_start_time = time.time()
    alpha_net, initial_marking, final_marking = alpha_miner.apply(event_log)
    alpha_execution_time = time.time() - alpha_start_time
    execution_times["alpha"] = alpha_execution_time
    print(f"Time to execute Alpha Miner: {alpha_execution_time:.3f}")

    # Inductive Miner (returns process tree)
    inductive_start_time = time.time()
    process_tree = inductive_miner.apply(event_log)
    # Convert to Petri net for unified analysis
    inductive_net, inductive_initial, inductive_final = pm4py.convert_to_petri_net(process_tree)
    inductive_execution_time = time.time() - inductive_start_time
    execution_times["inductive"] = inductive_execution_time
    print(f"Time to execute Inductive Miner: {inductive_execution_time:.3f}")

    # Heuristics Miner
    heuristics_start_time = time.time()
    heuristics_net, heuristics_initial, heuristics_final = heuristics_miner.apply(event_log)
    heuristics_execution_time = time.time() - heuristics_start_time
    execution_times["heuristics"] = heuristics_execution_time
    print(f"Time to execute Heuristics Miner: {heuristics_execution_time:.3f}")

    # ILP Miner
    ilp_start_time = time.time()
    ilp_net, ilp_initial, ilp_final = ilp_miner.apply(event_log)
    ilp_execution_time = time.time() - ilp_start_time
    execution_times["ilp"] = ilp_execution_time
    print(f"Time to execute ILP Miner: {ilp_execution_time:.3f}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create output file path and write to TXT file
    with open(os.path.join(output_dir, "execution_times.txt"), "w") as f:
        f.write("Algorithm Execution Times (seconds)\n")
        f.write("================================\n")
        for algorithm, exec_time in execution_times.items():
            f.write(f"{algorithm}: {exec_time:.3f}\n")

    print(f"Execution times saved to {os.path.join(output_dir, 'execution_times.txt')}")

    return {
        'alpha': (alpha_net, initial_marking, final_marking),
        'inductive': (inductive_net, inductive_initial, inductive_final),
        'heuristics': (heuristics_net, heuristics_initial, heuristics_final),
        'ilp': (ilp_net, ilp_initial, ilp_final)
    }


def calc_conformance(event_log, model, initial_marking, final_marking, algorithm_name):
    """
    Calculate conformance metrics Alignment-Based Fitness, Token-Based Precision, Alignment-Based Precision,
    Token-Based Precision, Token-Based Generalization.

    :param event_log: PM4Py event log
    :param model: Petri net model
    :param initial_marking: Initial marking
    :param final_marking: Final marking
    :param algorithm_name: Name of the algorithm

    :return dictionary: Dictionary containing algorithm name and calculated metrics
    """


    print(f"\n--- Conformance Analysis for {algorithm_name} ---")

    # Alignment-Based Fitness
    try:
        fitness_ab = fitness_alignments(event_log, model, initial_marking, final_marking)['log_fitness']
        print(f"Fitness (alignments-based): {fitness_ab:.4f}")
    except Exception as e:
        print(f"Error calculating fitness (alignments-based): {str(e)}")
        fitness_ab = None

    # Token-Based Fitness
    try:
        fitness_tbr = fitness_token_based_replay(event_log, model, initial_marking, final_marking)['log_fitness']
        print(f"Fitness (token-based replay): {fitness_tbr:.4f}")
    except Exception as e:
        print(f"Error calculating fitness (token-based replay): {str(e)}")
        fitness_tbr = None

    # Alignment-Based Precision
    try:
        precision_ab = precision_alignments(event_log, model, initial_marking, final_marking)
        print(f"Precision (alignments-based): {precision_ab:.4f}")
    except Exception as e:
        print(f"Error calculating precision (alignments-based): {str(e)}")
        precision_ab = None

    # Token-Based Precision
    try:
        precision_tbr = precision_token_based_replay(event_log, model, initial_marking, final_marking)
        print(f"Precision (token-based replay): {precision_tbr:.4f}")
    except Exception as e:
        print(f"Error calculating precision (token-based replay): {str(e)}")
        precision_tbr = None

    # Token-Based Generalization
    try:
        gen_tbr = generalization_tbr(event_log, model, initial_marking, final_marking)
        print(f"Generalization: {gen_tbr:.4f}")
    except Exception as e:
        print(f"Error calculating generalization: {str(e)}")
        gen_tbr = None

    metrics = {
        'algorithm': algorithm_name,
        'fitness_alignments': fitness_ab,
        'fitness_token_based': fitness_tbr,
        'precision_alignments': precision_ab,
        'precision_token_based': precision_tbr,
        'generalization': gen_tbr
    }

    return metrics


def save_conformance_to_csv(metrics_list, output_dir):
    """
    Save conformance metrics to a CSV file

    :param metrics_list: List of dictionaries containing conformance metrics
    :param output_dir: Path to the output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create output file path
    output_file = os.path.join(output_dir, "conformance_metrics.csv")

    # Define the field names (headers) for the CSV
    fieldnames = [
        'algorithm',
        'fitness_alignments',
        'fitness_token_based',
        'precision_alignments',
        'precision_token_based',
        'generalization'
    ]

    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in metrics_list:
            writer.writerow(metrics)

    print(f"Conformance metrics saved to {output_file}")


def export_pnml(petri_net, initial_marking, final_marking, algorithm_name, output_dir="output"):
    """
    Export the Petri net as a PNML file

    :param petri_net: Petri net model
    :param initial_marking: Initial marking
    :param final_marking: Final marking
    :param algorithm_name: Name of the algorithm
    :param output_dir: Path to the output directory

    :return output_file: Path to the output file
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create output file path
    output_file = os.path.join(output_dir, f"{algorithm_name}_model.pnml")

    # Export the Petri net as PNML file
    pm4py.write_pnml(petri_net, initial_marking, final_marking, output_file)

    print(f"PNML file exported: {output_file}")
    return output_file


def visualize_model(model, initial_marking=None, final_marking=None, algorithm_name=None, output_dir="output"):
    """
    Visualize the discovered process model

    :param model: Petri net model
    :param initial_marking: Initial marking
    :param final_marking: Final marking
    :param algorithm_name: Name of the algorithm
    :param process_tree: Process tree
    :param output_dir: Path to the output directory
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Visualize as Petri net
    parameters = {pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"}
    gviz = pn_visualizer.apply(model, initial_marking, final_marking, parameters=parameters)

    # Save to file
    pn_visualizer.save(gviz, os.path.join(output_dir, f"{algorithm_name}_petri_net.png"))

    print(f"Visualized petri net model saved to {os.path.join(output_dir, f"{algorithm_name}_petri_net.png")}")


def process_event_log(file_path, output_dir):
    """
    Process a single event log. Includes discovering process models, calculating conformance metrics and visualizing
    model.

    :param file_path: Path to the event log
    :param output_dir: Path to the output directory

    :return models: Discovered process models (not used, might be helpful for future work)
    :return pnml_files: Converted PNML files (not used, might be helpful for future work)
    """
    print(f"\n------------------------------------------------------")
    print(f"Processing event log: {file_path}")
    print(f"------------------------------------------------------")

    # Load the event log
    print(f"Loading event log...")
    event_log = load_event_log(file_path)
    print(f"Event log loaded: {len(event_log)} traces, {sum(len(trace) for trace in event_log)} events")

    # Discover models using the algorithms
    print("\nDiscovering process models...")
    models = discover_models(event_log, output_dir)
    print("\nModels discovered!")

    conformance_metrics = []

    # Process and export each model
    pnml_files = {}
    for algorithm_name, (model, initial_marking, final_marking) in models.items():
        # Calculate conformance metrics
        metrics = calc_conformance(event_log, model, initial_marking, final_marking, algorithm_name)
        conformance_metrics.append(metrics)

        # Visualize the model
        visualize_model(model, initial_marking, final_marking, algorithm_name, output_dir=output_dir)

        # Export as PNML
        pnml_file = export_pnml(model, initial_marking, final_marking, algorithm_name, output_dir=output_dir)
        pnml_files[algorithm_name] = pnml_file

    save_conformance_to_csv(conformance_metrics, output_dir)

    print(f"\nAnalysis for {os.path.basename(file_path)} complete!")

    return models, pnml_files


def main(input_folder, output_base_dir):
    """
    Main function to run the analysis on all event logs in a folder, processing them in alphabetical order by filename.

    :param input_folder: Path to the input folder
    :param output_base_dir: Path to the output folder

    :return results: Results of the processing (not used, might be helpful for future work)
    """

    print(f"Looking for event logs in {input_folder}...")

    # Get all XES files and sort them alphabetically by filename
    xes_files = []
    with os.scandir(input_folder) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".xes"):
                xes_files.append(entry)

    # Sort the files by name
    xes_files.sort(key=lambda x: x.name)

    print(f"Found {len(xes_files)} event log files. Processing in order:")
    for i, file_entry in enumerate(xes_files):
        print(f"{i + 1}. {file_entry.name}")

    results = {}

    # Process each file in order
    for file_entry in xes_files:
        output_directory = os.path.join(output_base_dir, file_entry.name)
        models, pnml_files = process_event_log(file_entry.path, output_directory)
        results[file_entry.name] = (models, pnml_files)

    print("\nAll event logs processed successfully!")
    return results


if __name__ == "__main__":

    # Specify input and output folders
    input_folder = "generated event logs/"
    output_base_dir = "output/"
    main(input_folder, output_base_dir)