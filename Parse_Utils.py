"""
========================================
Utility functions for data preprocessing
========================================

This module implements several utility functions for data preprocessing
It includes:
i) parse function for otu table file
ii) parse function for event file
iii) pick top k bacteria clusters
iV) SPIEC transform and its inverse
v) selecting specific events (e.g. antibiotic administration, bacteremia, etc)
"""

import numpy as np
import subprocess


def read_otu_table(file_name, cluster_method=None):
    """
    Parse the otu table file

    Parameters
    ----------
    file_name: String
        the name of the otuTable file
    cluster_method: a function mapping from a string to string
        maps each genus to its 'bacteria cluster'

    Returns
    -------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
        X[seq_idx][time_step][idx]' = the measurement count of idx^th bacteria cluster
        of the seq_idx sequence at time time_step
        None for missing measurements
    bacteria2idx: {String: int}
        a map from a 'bacteria cluster' to its corresponding dimension in a numpy array
    idx2bacteria: {int: String}
        a map from a dimension in the numpy array to its corresponding 'bacteria cluster'
    id2start_date: {int: int}
        a map from patientId (program id) to the starting measurement date
        relative to the reference date
    id2end_date: {int: int}
        a map from patientId (program id) to the ending measurement date
        relative to the reference date
    file_id2program_id: {int: int}
        a map from the actual patient id in the file to the index of X and U
    program_id2file_id: {int: int}
        a map from the index in X and U to the actual patient id in the original file
    """

    # sort the given otu file, create new ids for patients and store the new otu to a temp file
    print('cleaning data...')
    file_id2program_id, program_id2file_id = sort_otu(file_name, '.temp.tsv')
    print('reading otu table...')
    in_file = open('.temp.tsv', 'r')

    # initialize the variables
    line_count, bacteria2idx = 0, {}

    # default identity cluster method
    if cluster_method is None:
        cluster_method = lambda x: x

    # read the file
    for line in in_file:
        tokens = line.split('\t')

        # read the patient id
        if line_count == 0:
            assert (tokens[0] == 'patientId')
            patientId = [int(pid) for pid in tokens[1:]]

        # read the time
        elif line_count == 1:
            assert [tokens[0][:4] == 'time']
            time = [int(t) for t in tokens[1:]]
            assert (len(time) == len(patientId))
            X, idx2patient_date, id2start_date, id2end_date = initialize_X(patientId, time)

        # read the bacteria counts
        else:
            # record which (id, time_index) have been added
            # ignore repeats
            added_set = set()
            bacteria_name = tokens[0]
            new_bacteria = False
            key = cluster_method(bacteria_name)

            # add all "None" bacteria to "OTHER" cluster
            if key is None:
                key = 'OTHER'

            # create new cluster
            if bacteria2idx.get(key) is None:
                bacteria2idx[key] = len(bacteria2idx)
                new_bacteria = True
            bacteria_idx = bacteria2idx[key]
            bacteria_counts = tokens[1:]

            # read in the bacteria counts and add/create new entry for each array
            # (corresponding to one measurement)
            for idx in range(len(patientId)):
                pId, time_idx = idx2patient_date[idx]
                if (pId, time_idx) in added_set:
                    continue
                else:
                    added_set.add((pId, time_idx))
                bacteria_count = int(bacteria_counts[idx])
                if new_bacteria:
                    X[pId][time_idx].append(bacteria_count)
                    assert (len(X[pId][time_idx]) == len(bacteria2idx))
                else:
                    X[pId][time_idx][bacteria_idx] += bacteria_count

        line_count += 1

    # close the reading io file and delete the temp file
    in_file.close()
    subprocess.call(['rm', '-rf', '.temp.tsv'])

    # make each of the measurement array a numpy array
    X = [[(np.array(x) if x is not None else x) for x in xs] for xs in X]

    # map the dimension of a numpy array (for one measurement) to its corresponding bacteria cluster name
    idx2bacteria = dict([(bacteria2idx[key], key) for key in bacteria2idx])

    return X, bacteria2idx, idx2bacteria, id2start_date, id2end_date, file_id2program_id, program_id2file_id


def initialize_X(patientId, time):
    """
    Initialize the returned X, whose format is suitable for the Kalman Filter program

    Parameters
    ----------
    patientId: [num_measurements] array of int
        the array of patientId (program id)
    time: [num_measurements] array of int
        the array of measurement data relative to the reference

    Returns
    -------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
        X[seq_idx][time_step][idx]' = the measurement count of idx^th bacteria cluster
        of the seq_idx sequence at time time_step
        None for missing measurements
    id2patient_date: {int: (int, int)}
        a map from measurement index [0, measurement_count) to (patient_id, measurement_date)
    id2start_date: {int: int}
        a map from patientId (program id) to the starting measurement date
        relative to the reference date
    id2end_date: {int: int}
        a map from patientId (program id) to the ending measurement date
        relative to the reference date
    """
    # the starting and ending measurement date
    id2start_date, id2end_date = {}, {}
    for idx in range(len(patientId)):
        pId = patientId[idx]
        if id2start_date.get(pId) is None:
            id2start_date[pId] = time[idx]
        id2end_date[pId] = time[idx]
    patient_count = len(id2start_date)

    # initialize X, [] for days that have measurements, None for missing measurements
    X = [[None for _ in range(id2end_date[pId] - id2start_date[pId] + 1)]
         for pId in range(patient_count)]
    idx2patient_date = {}
    for idx in range(len(patientId)):
        pId, time_idx = patientId[idx], time[idx] - id2start_date[patientId[idx]]
        idx2patient_date[idx] = (pId, time_idx)
        X[pId][time_idx] = []

    return X, idx2patient_date, id2start_date, id2end_date


def pick_topk(X, idx2bacteria, k=10, includes=None, excludes=None, include_other=True):
    """
    Picking the top k clusters, including/excluding certain clusters
    add all the remaining clusters to the 'OTHER' cluster

    Parameters
    ----------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
        X[seq_idx][time_step][idx]' = the measurement count of idx^th bacteria cluster
        of the seq_idx sequence at time time_step
        None for missing measurements
    idx2bacteria: {int: String}
        a map from a dimension in the numpy array to its corresponding 'bacteria cluster'
    k: int
       number of top (in terms of relative abundance) clusters to pick, in addition to "include" argument
    includes: array of Strings
        an array of 'bacteria clusters' to be included (in addition to the top k clusters)
    excludes: array of Strings
        an array of bacteria clusters to be excluded
    include_other: boolean
         whether to include the 'OTHER' cluster in the returned X

    Returns
    -------
    returned_idx2bacteria: {int: String}
        after selecting the top k 'bacteria cluster',
        the resulting map from a dimension in the numpy array to its corresponding 'bacteria cluster'
    returned_X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
        after selecting the top k 'bacteria cluster'
        returned_X[seq_idx][time_step][idx]' = the measurement count of idx^th bacteria cluster
        of the seq_idx sequence at time time_step
        None for missing measurements
    """

    includes, excludes = [] if includes is None else includes, [] if excludes is None else excludes

    # raise exceptions if arguments do not meet requirements
    valid_args_pick_topk(X, idx2bacteria, k, includes, excludes, include_other)

    # create an ordering of bacteria clusters according to their average srelative abundance
    sorted_idx, sorted_bacteria = create_order(X, idx2bacteria)

    # pick a list of dimensions that will appear in returned_X
    picked_dimensions = pick_dimension_list(idx2bacteria, sorted_idx, k, includes, excludes)

    # creating the returned bacteria dict
    returned_idx2bacteria = [idx2bacteria[idx] for idx in picked_dimensions]
    if include_other:
        returned_idx2bacteria.append('OTHER')

    # pick the dimension for each measurements in the raw counts
    returned_X = create_returned_X(X, picked_dimensions, include_other)

    return returned_X, returned_idx2bacteria


def valid_args_pick_topk(X, idx2bacteria, k, includes, excludes, include_other):
    """
    Confirm that the arguments for pick_topk meet the requirements
    Arguments are the same with pick_topk function
    """

    # all possible bacteria clusters name
    all_bacteria = [idx2bacteria[key] for key in idx2bacteria]

    for i in includes:
        # ensure that includes and excludes do not overlap
        if i in excludes:
            raise Exception('argument \"includes\" cannot overlap with \"excludes\"')
        # ensure that all clusters in includes are in idx2bacteria
        if i not in all_bacteria:
            raise Exception('%s is not in idx2bacteria' % i)
    if 'OTHER' in includes:
        raise Exception('OTHER is not a valide bacteria cluster name')


def create_order(X, idx2bacteria):
    """
    Create an order of bacteria (to pick the top) and its correponding index

    Parameters
    ----------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
        X[seq_idx][time_step][idx]' = the measurement count of idx^th bacteria cluster
        of the seq_idx sequence at time time_step
        None for missing measurements
    idx2bacteria: {int: String}
        a map from a dimension in the numpy array to its corresponding 'bacteria cluster'

    Returns
    -------
    sorted_idx: array of int
        sorted (according to average relative abundance) index of the 'bacteria cluster'
    sorted_bacteria: array of String
        sorted (according to average relative abundance) 'bacteria cluster' name
    """
    total_abundance = np.zeros(len(idx2bacteria))
    for seq_idx in range(len(X)):
        for time_step in range(len(X[seq_idx])):
            if X[seq_idx][time_step] is not None:
                total_abundance += X[seq_idx][time_step] / np.sum(X[seq_idx][time_step])
    # sort the array by descending order
    sorted_idx = np.argsort(-total_abundance)
    sorted_bacteria = [idx2bacteria[idx] for idx in sorted_idx]
    return sorted_idx, sorted_bacteria


def pick_dimension_list(idx2bacteria, sorted_idx, k, includes, excludes):
    """
    Pick the dimensions of the bacteria that meets the requirement (top k or in includes, not in excludes)

    Parameters
    ----------
    idx2bacteria: {int: String}
        a map from a dimension in the numpy array to its corresponding 'bacteria cluster'
    sorted_idx: array of int
        sorted (according to average relative abundance) index of the 'bacteria cluster'
    k: int
        number of top (in terms of relative abundance) clusters to pick, in addition to "include" argument
    includes: array of String
        an array of bacteria clusters to be included (in addition to the top k clusters)
    excludes: array of String
        an array of bacteria clusters to be excluded
    Returns
    -------
    picked_dimensions: an array of int
        a list of index corresponding to the top bacteria clusters
    """
    remaining_top_clusters = k
    picked_dimensions = []
    for idx in sorted_idx:
        if idx2bacteria[idx] in includes:
            picked_dimensions.append(idx)
        elif idx2bacteria[idx] in excludes or idx2bacteria[idx] == 'OTHER':
            pass
        elif remaining_top_clusters > 0:
            picked_dimensions.append(idx)
            remaining_top_clusters -= 1
        else:
            pass
    return picked_dimensions


def create_returned_X(X, picked_dimensions, include_other):
    """
    Create the returned X, given the picked dimensions

    Parameters
    ----------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
        X[seq_idx][time_step][idx]' = the measurement count of idx^th bacteria cluster
        of the seq_idx sequence at time time_step
        None for missing measurements
    picked_dimension: an array of int
        a list of index corresponding to the top bacteria clusters
    include_other: boolean
         whether to include the 'OTHER' cluster in the returned X

    Returns
    -------
    returned_X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
        after selecting the top k 'bacteria cluster'
        returned_X[seq_idx][time_step][idx]' = the measurement count of idx^th bacteria cluster
        of the seq_idx sequence at time time_step
        None for missing measurements
    """
    returned_X = []
    for x_seq in X:
        returned_x_seq = []
        for x in x_seq:
            # append missing measurements
            if x is None:
                returned_x_seq.append(None)
            else:
                # pick the chosen dimensions
                new_x = x[picked_dimensions]
                # add all other entries to "OTHER" cluster
                if include_other:
                    new_x = np.append(new_x, [np.sum(x) - np.sum(new_x)])
                returned_x_seq.append(new_x)
        returned_X.append(returned_x_seq)
    return returned_X


def measurement_mean(x, epsilon=0):
    """
    Transform counts to (smoothed) frequencies (relative abundance)

    Parameters
    ----------
    x: [dimension] numpy array
        measurement
    epsilon: float
        the smoothing factor. default 0 - equivalent to w\ out smoothing

    Returns
    -------
    x / np.sum(x): the smoothed frequency of x
    """
    x += epsilon
    return x / np.sum(x)


def SPIEC_transform(x):
    """
    SPIEC transform

    Parameters
    ----------
    x: [dimension] numpy array
        measurement frequency (all entries add up to 1)

    Returns
    -------
    y: [dimension - 1] numpy array
        y = spiec(x)
        y_i = log(x_{i} / geometric_mean(x)), except the last dimension;
        last dimension is dropped
    """
    # ensure that the measurement is a frequency
    assert (np.abs(np.sum(x) - 1) < 1e-7)
    y = np.log(x)
    y -= np.mean(y)
    return y[:-1]


def inverse_SPIEC(y):
    """
    Inverse of SPIEC transform

    Parameters
    ----------
    y: [dimension] numpy array

    Returns
    -------
    x : [dimension + 1] numpy array
        x = spiec^{-1}(y)
    """
    x = np.append(y, [1])
    x = np.exp(x)
    x = x / np.sum(x)
    return x


def default_transform(x):
    """
    Default transformation

    Parameters
    ----------
    x: [dimension] array of int
        vector representing measurement count

    Returns
    -------
    x: [dimension - 1] numpy array
        defualt transformation of mean with smoothed factor=1 followed by SPIEC
    """
    x = measurement_mean(x, epsilon=1)
    x = SPIEC_transform(x)
    return x


def map_2D_array(X, f):
    """
    A wrapper that maps each measurement/control/events/arrays to an element,
    while perserving the missing measurements and the same data structure

    Parameters
    ----------
    X: [seq_count] array of [seq_length] array of (numpy array/None)
    f: a function that maps from a numpy vector/None to an element

    Returns
    -------
    returned_X: [seq_count] array of [seq_length] array of Objects
    """
    returned_X = [[f(x) if x is not None else None for x in x_seq] for x_seq in X]
    return returned_X


def default_measurement_transformation(X):
    """
    Apply default measurement transformation to measurement count X
    """
    return map_2D_array(X, default_transform)


def read_event(file_name, id2start_date, id2end_date, file_id2program_id):
    """
    Parse the event file

    Parameters
    ----------
    file_name: String
        the name of the event file
    id2start_date: {int: int}
        a map from patientId (program id) to the starting measurement date
        relative to the reference date
    id2end_date: {int: int}
        a map from patientId (program id) to the ending measurement date
        relative to the reference date
    file_id2program_id: {int: int}
        a map from the actual patient id in the file to the index of X and U
    Returns
    -------
    U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
        one-hot encoded
        'U[seq_idx][time_step][idx]' = 1 if idx^{th} event happens in the seq_idx sequence at time time_step
    event_name2idx: {(String, String): int}
        a map from (type, description) to dimension in each of the numpy array in U
    idx2event_name: {int: (String, String)}
        a map from dimension in each of the numpy array in U to (type, description)
    file_id2program_id: {int: int}
        a map from the actual patient id in the file to the index of X and U
    """
    # get the mapping from event_name to index
    print('reading event table...')
    event_name2idx = get_event_name2idx(file_name)
    # get the reverse mapping from index to event_name
    idx2event_name = dict([(event_name2idx[event_name], event_name) for event_name in event_name2idx])
    event_count = len(event_name2idx)

    # initialize U
    U = initialize_U(id2start_date, id2end_date, event_count)

    # adding the events to U
    in_file = open(file_name, 'r')
    for l in in_file:
        add_event(l, U, id2start_date, event_name2idx, file_id2program_id)

    in_file.close()

    return U, event_name2idx, idx2event_name

def add_event(l, U, id2start_date, event_name2idx, file_id2program_id):
    """
    Read a line of event description in the event file and add it to U

    Parameters
    ----------
    l: String
        a line in the event description, typically of the format
        <patient_id>\t<type>\t<description>\t<start_date>\t<end_date>
    U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
        one-hot encoded
        'U[seq_idx][time_step][idx]' = 1 if idx^{th} event happens in the seq_idx sequence at time time_step
    id2start_date: {int: int}
        a map from patientId (program id) to the starting measurement date
        relative to the reference date
    event_name2idx: {(String, String): int}
        a map from (type, description) to dimension in each of the numpy array in U
    file_id2program_id: {int: int}
        a map from the actual patient id in the file to the index of X and U
    """
    try:
        id, event_name, start, end = parse_line_to_event(l, file_id2program_id)
        start_idx = max(0, start - id2start_date[id])
        end_idx = min(len(U[id]), end + 1 - id2start_date[id])
        for time_step in range(start_idx, end_idx):
            U[id][time_step][event_name2idx[event_name]] = 1
    except:
        return


def initialize_U(id2start_date, id2end_date, event_count):
    """
    initialize the returned U array
    """
    U = [np.zeros((id2end_date[id] - id2start_date[id] + 1, event_count))
         for id in id2start_date]
    return U


def get_event_name2idx(file_name):
    """
    Read the event file and create a dict that maps the event name to an index

    Parameters
    ----------
    file_name: String
        the name of the event file

    Returns
    -------
    event_name2idx: {(String, String): int}
        a map from (type, description) to dimension in each of the numpy array in U
    """
    in_file = open(file_name, 'r')
    event_name2idx = {}
    for l in in_file:
        try:
            _, event_name, _, _ = parse_line_to_event(l, {})
            if event_name2idx.get(event_name) is None:
                event_name2idx[event_name] = len(event_name2idx)
        except Exception as e:
            print(e)
            pass
    in_file.close()
    return event_name2idx


def parse_line_to_event(line, file_id2program_id):
    """
    Parse a line of event description to a list of key elements of the event

    Parameters
    ----------
    line: String
        a line in the event description, typically of the format
        <patient_id>\t<type>\t<description>\t<start_date>\t<end_date>
    file_id2program_id: {int: int}
        a map from the actual patient id in the file to the index of X and U
    Returns
    -------
    id: int
        patient id program id
    event_name: (String, String)
        (type, description). e.g. ('Antibiotic', 'Vancomycin')
    start: int
        the start date of the event (inclusive) relative to the reference date
    end: int
        the end date of the event (inclusive) relative to the reference date
    """
    id, type, description, start, end = line.split('\t')
    id = file_id2program_id.get(int(id))
    event_name = (type, description)
    start = int(start)
    end = int(end)
    return id, event_name, start, end


def pick_events_by_dims(U, picked_dimensions):
    """
    Pick the selected events of U according to the index they correspond

    Parameters
    ----------
    U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
        one-hot encoded
        'U[seq_idx][time_step][idx]' = 1 if idx^{th} event happens in the seq_idx sequence at time time_step
    picked_dimensions: array of int
        a list of index corresponding to the selected events

    Returns
    -------
    returned_U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
        one-hot encoded
    """
    returned_U = [[u[picked_dimensions] if u is not None else None for u in u_seq] for u_seq in U]
    return returned_U


def extract_event_by_filter(U, idx2event_name, filter):
    """
    Pick the selected events of U according to the filter function

    Parameters
    ----------
    U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
        one-hot encoded
        'U[seq_idx][time_step][idx]' = 1 if idx^{th} event happens in the seq_idx sequence at time time_step
    idx2event_name: {int: (String, String)}
        a map from dimension in each of the numpy array in U to (type, description)
    filter: a function mapping from (String, String) to boolean
        whether to select the event (type, description) in the returned U

    Returns
    -------
    returned_U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
        one-hot encoded
        after selecting a type of event 'U[seq_idx][time_step][idx]' = 1
        if idx^{th} event happens in the seq_idx sequence at time time_step
    returned_idx2event_name: array of (String, String)
        a 'map' from dimension in each of the numpy array in returned_U to (type, description)
    """
    returned_idx2event_name, picked_dimensions = {}, []
    for idx in idx2event_name:
        event_name = idx2event_name[idx]
        if filter(event_name):
            returned_idx2event_name[len(returned_idx2event_name)] = event_name
            picked_dimensions.append(idx)
    returned_U = pick_events_by_dims(U, picked_dimensions)
    return returned_U, returned_idx2event_name


def extract_event_by_type(U, idx2event_name, type):
    """
    Picking the events that has the specified "type" (e.g. "Antibiotic", "Bacteremia", etc)

    Parameters
    ----------
    U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
        one-hot encoded
        'U[seq_idx][time_step][idx]' = 1 if idx^{th} event happens in the seq_idx sequence at time time_step
    idx2event_name: {int: (String, String)}
        a map from dimension in each of the numpy array in U to (type, description)
    type: String
        the type to be selected, e.g. 'Antibiotic', 'Bacteremia'

    Returns
    -------
    returned_U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
        one-hot encoded
        after selecting a type of event 'U[seq_idx][time_step][idx]' = 1
        if idx^{th} event happens in the seq_idx sequence at time time_step
    returned_idx2event_name: array of (String, String)
        a 'map' from dimension in each of the numpy array in returned_U to (type, description)
    """

    def filter(event_name):
        try:
            return event_name[0] == type
        except:
            return False

    returned_U, returned_idx2event_name = extract_event_by_filter(U, idx2event_name, filter)
    return returned_U, returned_idx2event_name


def or_U(U):
    """
    For one patient, one time_step, if one entry of u is 1, it is considered 1
    """
    return map_2D_array(U, lambda x: x.any())


def parse_genus_file(arg_file):
    """
    Parse the file that include the genuses name (one every line)

    Parameters
    ----------
    arg_file: array of String of size 0 or None
        if None, then arg_file is not specified
        otherwise, the first element is the file name
        the file containing names of genuses, one every line
    Returns
    -------
    genuses: an array of String
        each string in teh array is a genus to be included
    """
    if arg_file is None:
        return []
    file_name = arg_file[0]
    in_file = open(file_name, 'r')
    genuses = []
    for l in in_file:
        genuses.append(l.strip())
    in_file.close()
    return genuses

def permute_array(original, permutation):
    """
    Permute the original array under the permutation

    Parameters
    ----------
    original: array
        the original array to be permuted
    permutation: int array
        a permutation of [0, len(array))
    Returns
    -------
    permuted: array
        the permuted array of the original under permutation
    """
    num_elements = len(permutation)
    assert(len(original) == num_elements)
    permuted = [0] * num_elements
    for idx in range(num_elements):
        permuted[idx] = original[permutation[idx]]
    return permuted

def sort_otu(otu_file, sorted_file):
    """
    Parse an otu table file, sort it, and write it to a temp file

    Parameters
    ----------
    otu_file: String
        input otu table file name
    sorted_file: String
        output otu table file name
    Returns
    -------
    file_id2program_id: {int: int}
        a map from the actual patient id in the file to the index of X and U
    program_id2file_id: {int: int}
        a map from the index in X and U to the actual patient id in the original file
    """
    # get the patient id and time and create program_ids for the patients
    in_file = open(otu_file, 'r')
    patientIds, time = [int(_) for _  in in_file.readline().split('\t')[1:]], [int(_) for _ in in_file.readline().split('\t')[1:]]
    num_measurements = len(patientIds)
    file_id2program_id, program_id2file_id = create_id(patientIds)

    # sort according to program_id (first) and date (second)
    permutation = sorted(range(num_measurements), key=lambda idx: (file_id2program_id[patientIds[idx]], time[idx]))

    out_file = open(sorted_file, 'w')
    # write the line representing id
    id_line = '\t'.join(['patientId']+ [str(file_id2program_id[patientIds[permutation[idx]]])
                                        for idx in range(num_measurements)])+ '\n'
    out_file.write(id_line)
    # write the line representing measurement time (relative to measurement day)
    time_line = '\t'.join(['time'] + [str(time[permutation[idx]]) for idx in range(num_measurements)]) + '\n'
    out_file.write(time_line)

    # write the bacteria count line
    for l in in_file:
        tokens = l.split('\t')
        bacteria_line = tokens[0] + '\t'
        bacteria_line += '\t'.join([tokens[1:][permutation[idx]] for idx in range(num_measurements)])
        out_file.write(bacteria_line)

    # close the files
    in_file.close()
    out_file.close()
    return file_id2program_id, program_id2file_id


def create_id(patientIds):
    """
    Read the patientIds
    Create program id (index for X and U) for every patient id

    Parameters
    ----------
    patientIds: an array of String
        patient ids (the first line of the otu table file)
    Returns
    -------
    file_id2program_id: {int: int}
        a map from the actual patient id in the file to the index of X and U
    program_id2file_id: {int: int}
        a map from the index in X and U to the actual patient id in the original file
    """
    file_id2program_id, program_id2file_id = {}, {}
    for file_id in patientIds:
        if file_id2program_id.get(file_id) is None:
            file_id2program_id[file_id] = len(file_id2program_id)
            program_id2file_id[len(program_id2file_id)] = file_id
    return file_id2program_id, program_id2file_id