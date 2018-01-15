"""
=======================================
Kalman Filter inference for microbioime
=======================================

This module implements a command line type function
that applies Kalman Filter algorithms to microbiome data
"""

import argparse
from Parse_Utils import read_otu_table, read_event, map_2D_array, inverse_SPIEC,\
    pick_topk, extract_event_by_type, default_measurement_transformation, parse_genus_file
from Utils import write_otu
from Kalman_Filter import Kalman_Filter
import matplotlib.pyplot as plt
import subprocess

if __name__ == "__main__":
    # parsing the arguments
    # use command 'python ubiome-kf.py --help' to learn details
    parser = argparse.ArgumentParser(description='Applying Kalman Filter to Microbiome')
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--otu_file', nargs=1, type=str, required=True, help='otu table file name')
    required_args.add_argument('--event_file', nargs=1, type=str, required=True, help='event table file name')
    optional_args = parser.add_argument_group('optional arguments')
    optional_args.add_argument('--output_dir', nargs=1, type=str, help='the output directory of the kalman filter',
                               default='./')
    optional_args.add_argument('--k', nargs=1, type=int, help='the top number of clusters to pick',
                               default=10)
    optional_args.add_argument('--includes', nargs=1, type=str, help='the file containing bacteria genuses to include',
                               default=None)
    optional_args.add_argument('--excludes', nargs=1, type=str, help='the file containing bacteria genuses to exclude',
                               default=None)
    optional_args.add_argument('--include_other', nargs=1, type=bool, help='whether to add all excluded clusters to '
                                                                           '\"OTHER\" cluster',
                               default=True)
    optional_args.add_argument('--epsilon', nargs=1, type=float, help='the convergenece threshold',
                               default=1e-4)
    optional_args.add_argument('--max_iteration', nargs=1, type=int, help='the maximum iterations of EM to run',
                               default=1000)
    optional_args.add_argument('--print_every', nargs=1, type=int, help='print every print_every iterations in EM',
                               default=20)
    args = parser.parse_args()

    # read the otu table file and preprocess
    measurements, bacteria2idx, idx2bacteria, id2start_date, id2end_date = read_otu_table(args.otu_file[0])
    includes, excludes = parse_genus_file(args.includes[0]), parse_genus_file(args.excludes[0])
    measurements, idx2bacteria = pick_topk(measurements, idx2bacteria, k=args.k[0], includes=includes,
                                           excludes=excludes, include_other=args.include_other[0])
    measurements = default_measurement_transformation(measurements)
    print('Finish reading the otu table file')

    # read the event file and preprocess
    U, event_name2idx, idx2event_name = read_event(args.event_file[0], id2start_date, id2end_date)
    antibiotic_history, antibiotic_dict = extract_event_by_type(U, idx2event_name, 'Antibiotic')
    print('Finish reading the event file')

    # setting up EM
    dimension, control_dimension = len(idx2bacteria) - 1, len(antibiotic_dict)
    kf = Kalman_Filter(dimension, control_dimension)
    kf.config['threshold'] = args.epsilon
    kf.config['num_iterations'] = args.max_iteration[0]

    # training
    print('Training')
    kf.fit(measurements, antibiotic_history, print_every=args.print_every[0])

    # inferring
    predicted, filtered, smoothed = kf.estimate(measurements, antibiotic_history)

    # transform back
    predicted_mean_rel_abundance = map_2D_array(predicted['mean'], inverse_SPIEC)
    filtered_mean_rel_abundance = map_2D_array(filtered['mean'], inverse_SPIEC)
    smoothed_mean_rel_abundance = map_2D_array(smoothed['mean'], inverse_SPIEC)

    # make the output_dir a folder
    output_dir = args.output_dir[0]
    if output_dir[-1] != '/':
        output_dir += '/'

    # create the output directory
    subprocess.call(['rm', '-rf', output_dir])
    subprocess.call(['mkdir', output_dir])

    # plot the log likelihood over iterations
    plt.plot(kf.ll_history)
    plt.title('EM Optimization')
    plt.xlabel('iteration')
    plt.ylabel('log likelihood per measurement')
    plt.savefig(output_dir + 'EM_log_likelihood.png')

    # write the inferred results in otu table format
    write_otu(predicted_mean_rel_abundance, output_dir + 'predicted.tsv', idx2bacteria, id2start_date)
    write_otu(filtered_mean_rel_abundance, output_dir + 'filtered.tsv', idx2bacteria, id2start_date)
    write_otu(smoothed_mean_rel_abundance, output_dir + 'smoothed.tsv', idx2bacteria, id2start_date)