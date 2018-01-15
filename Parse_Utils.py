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


def map_name_to_genus(s):
    """
    Map a complete name of a bacteria to its genus name; default clustering method of read_otu_table

    Parameters
    ----------
    s: String
        complete bacteria name of format "<Phylum>;<Class>;<Order>;<Family>;<Genus>"
        e.g. Firmicutes;Bacilli;Lactobacillales;Enterococcaceae;Enterococcus

    Returns
    -------
    s.split(";")[-1]: String
        <Genus>
    """
    return s.split(";")[-1]


def read_otu_table(file_name, cluster_method=map_name_to_genus):
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
        a map from patientId (starting from 0) to the starting measurement date
        relative to the reference date
    id2end_date: {int: int}
        a map from patientId (starting from 0) to the ending measurement date
        relative to the reference date
    """
    in_file = open(file_name, 'r')

    # initialize the variables
    line_count, bacteria2idx = 0, {}

    # read the file
    for line in in_file:
        tokens = line.split('\t')

        # read the patient id
        if line_count == 0:
            assert (tokens[0] == 'patientId')
            patientId = tokens[1:]

            # patientId starts from 1 in the file
            # starts from 0 in this program
            patientId = [int(pid) - 1 for pid in patientId]

        # read the time
        elif line_count == 1:
            assert [tokens[0][:4] == 'time']
            time = [int(t) for t in tokens[1:]]
            assert (len(time) == len(patientId))
            X, idx2patient_date, id2start_date, id2end_date = initialize_X(patientId, time)

        # read the bacteria counts
        else:
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

            # read in the bacteria counts and add/create new entry for each array (corresponding to)
            # one measurement
            for idx in range(len(patientId)):
                pId, time_idx = idx2patient_date[idx]
                bacteria_count = int(bacteria_counts[idx])
                if new_bacteria:
                    X[pId][time_idx].append(bacteria_count)
                    assert (len(X[pId][time_idx]) == len(bacteria2idx))
                else:
                    X[pId][time_idx][bacteria_idx] += bacteria_count

        line_count += 1

    # make each of the measurement array a numpy array
    X = [[(np.array(x) if x is not None else x) for x in xs] for xs in X]

    # close the reading io file
    in_file.close()

    # map the dimension of a numpy array (for one measurement) to its corresponding bacteria cluster name
    idx2bacteria = dict([(bacteria2idx[key], key) for key in bacteria2idx])

    return X, bacteria2idx, idx2bacteria, id2start_date, id2end_date


def initialize_X(patientId, time):
    """
    Initialize the returned X, whose format is suitable for the Kalman Filter program

    Parameters
    ----------
    patientId: [num_measurements] array of int
        the array of patientId, starting from 0
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
        a map from patientId (starting from 0) to the starting measurement date
        relative to the reference date
    id2end_date: {int: int}
        a map from patientId (starting from 0) to the ending measurement date
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
    X = [[None for time_step in range(id2end_date[pId] - id2start_date[pId] + 1)]
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


def read_event(file_name, id2start_date, id2end_date):
    """
    Parse the event file

    Parameters
    ----------
    file_name: String
        the name of the event file
    id2start_date: {int: int}
        a map from patientId (starting from 0) to the starting measurement date
        relative to the reference date
    id2end_date: {int: int}
        a map from patientId (starting from 0) to the ending measurement date
        relative to the reference date

    Returns
    -------
    U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
        one-hot encoded
        'U[seq_idx][time_step][idx]' = 1 if idx^{th} event happens in the seq_idx sequence at time time_step
    event_name2idx: {(String, String): int}
        a map from (type, description) to dimension in each of the numpy array in U
    idx2event_name: {int: (String, String)}
        a map from dimension in each of the numpy array in U to (type, description)
    """
    # get the mapping from event_name to index
    event_name2idx = get_event_name2idx(file_name)
    # get the reverse mapping from index to event_name
    idx2event_name = dict([(event_name2idx[event_name], event_name) for event_name in event_name2idx])
    event_count = len(event_name2idx)

    # initialize U
    U = initialize_U(id2start_date, id2end_date, event_count)

    # adding the events to U
    in_file = open(file_name, 'r')
    for l in in_file:
        add_event(l, U, id2start_date, id2end_date, event_name2idx)

    in_file.close()

    return U, event_name2idx, idx2event_name


def add_event(l, U, id2start_date, id2end_date, event_name2idx):
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
        a map from patientId (starting from 0) to the starting measurement date
        relative to the reference date
    id2end_date: {int: int}
        a map from patientId (starting from 0) to the ending measurement date
        relative to the reference date
    event_name2idx: {(String, String): int}
        a map from (type, description) to dimension in each of the numpy array in U
    """
    try:
        id, event_name, start, end = parse_line_to_event(l)
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
            _, event_name, _, _ = parse_line_to_event(l)
            if event_name2idx.get(event_name) is None:
                event_name2idx[event_name] = len(event_name2idx)
        except:
            pass
    in_file.close()
    return event_name2idx


def parse_line_to_event(line):
    """
    Parse a line of event description to a list of key elements of the event

    Parameters
    ----------
    line: String
        a line in the event description, typically of the format
        <patient_id>\t<type>\t<description>\t<start_date>\t<end_date>

    Returns
    -------
    id: int
        patient id (starting from 0)
    event_name: (String, String)
        (type, description). e.g. ('Antibiotic', 'Vancomycin')
    start: int
        the start date of the event (inclusive) relative to the reference date
    end: int
        the end date of the event (inclusive) relative to the reference date
    """
    id, type, description, start, end = line.split('\t')
    id = int(id) - 1
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


def parse_genus_file(file_name):
    """

    Parametersa
    ----------
    file_name: String
        the file containing names of genuses, one every line
    Returns
    -------
    genuses: an array of String
        each string in teh array is a genus to be included
    """
    if file_name is None:
        return []
    in_file = open(file_name, 'r')
    genuses = []
    for l in in_file:
        genuses.append(l.strip())
    in_file.close()
    return genuses