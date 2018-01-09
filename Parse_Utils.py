import numpy as np

'''
    Map a complete name of a bacteria to its genus name; default clustering method of read_otu_table
    
    # Arguments
        s:complete bacteria name of format "<Phylum>;<Class>;<Order>;<Family>;<Genus>"
            e.g. Firmicutes;Bacilli;Lactobacillales;Enterococcaceae;Enterococcus
    # Returns
        s.split(";")[-1]: <Genus>
'''
def map_name_to_genus(s):
    return s.split(";")[-1]

'''
    Parse the otu table file
    
    # Arguments
        file_name: the name of the otuTable file
        cluster_method: a function that maps each genus to its cluster;
            default cluster by genus (map_name_to_genus)
    # Returns
        X: an array of (length=number of patients) array of (length=number of measurement span) of
            numpy array (shape=(number_of_bacteria_clusters,)) or None values (for missing) measurements
        bacteria2idx: a map from a bacteria cluster to its corresponding dimension in a numpy array
        idx2bacteria: a map from a dimension in the numpy array to its corresponding bacteria cluster
        id2start_date: a map from patientId (starting from 0) to the starting measurement date
            relative to the reference date
        id2end_date: a map from patientId (starting from 0) to the ending measurement date
            relative to the reference date
'''
def read_otu_table(file_name, cluster_method=map_name_to_genus):
    in_file = open(file_name, 'r')
    
    # initialize the variables
    line_count, bacteria2idx = 0, {}
    
    # read the file
    for line in in_file:
        tokens = line.split('\t')
        
        # read the patient id
        if line_count == 0:
            assert(tokens[0] == 'patientId')
            patientId = tokens[1:]
            
            # patientId starts from 1 in the file
            # starts from 0 in this program
            patientId = [int(pid) - 1 for pid in patientId]
        
        # read the time
        elif line_count == 1:
            assert[tokens[0][:4] == 'time']
            time = [int(t) for t in tokens[1:]]
            assert(len(time) == len(patientId))
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
                    assert(len(X[pId][time_idx]) == len(bacteria2idx))
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

'''
    Initialize the returned X, whose format is suitable for the Kalman Filter program
    
    # Arguments
        patientId: the array of patientId, starting from 0
        time: the array of measurement data relative to the reference
    # Returns
        X: an array of (length=number of patients) array of (length=number of measurement span) of
            empty array or None values (for missing) measurements
        idx2patient_date: a map from measurement index to (patient_id, measurement_date)
        id2start_date: a map from patient id to his/her starting measurement date relative to the reference date
        id2end_date: a map from patient id to his/her last measurement date relative to the reference date
'''
def initialize_X(patientId, time):
    
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

'''
    Picking the top k clusters, including/excluding certain clusters; add all the remaining clusters to the 'OTHER' cluster
    
    # Arguments
        X: an array of (length=number of patients) array of (length=number of measurement span) of
            numpy array (shape=(number_of_bacteria_clusters,)) or None values (for missing) measurements
        bacteria2idx: a map from a bacteria cluster to its corresponding dimension in a numpy array
        idx2bacteria: a map from a dimension in the numpy array to its corresponding bacteria cluster
        k: number of top (in terms of relative abundance) clusters to pick, in addition to "include" argument
        includes: an array of bacteria clusters to be included (in addition to the top k clusters)
        excludes: an array of bacteria clusters to be excluded
        include_other: whether to include the 'OTHER' cluster in the returned X
        logically, includes and excludes cannot have overlap
    # Returns
        returned_idx2bacteria: a map from a dimension in the numpy array to its corresponding bacteria cluster
            in returned_X
        returned_X: an array of (length=number of patients) array of (length=number of measurement span) of
            numpy array (shape=(number_of_bacteria_clusters,)) or None values (for missing) measurements;
            still, each of the measurements consists of counts, not relative abundance
'''
def pick_topk(X, idx2bacteria, k=10, includes=None, excludes=None, include_other=True):
    
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

'''
    Confirm that the arguments for pick_topk meet the requirements
    
    # Arguments
        exactly the same with pick_topk
'''
def valid_args_pick_topk(X, idx2bacteria, k, includes, excludes, include_other):
    
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

'''
    Create an order of bacteria (to pick the top) and its correponding index
    
    # Arguments
        X: an array of (length=number of patients) array of (length=number of measurement span) of
            numpy array (shape=(number_of_bacteria_clusters,)) or None values (for missing) measurements;
            each of the measurement can be raw counts (or frequencies)
        idx2bacteria: a map from a dimension in the numpy array to its corresponding bacteria cluster
            in returned_X
    # Returns
        sorted_idx: sorted (according to average relative abundance) index of the bacteria cluster
        sorted_bacteria: sorted (according to average relative abundance) bacteria cluster name
'''
def create_order(X, idx2bacteria):
    total_abundance = np.zeros(len(idx2bacteria))
    for seq_idx in range(len(X)):
        for time_step in range(len(X[seq_idx])):
            if X[seq_idx][time_step] is not None:
                total_abundance += X[seq_idx][time_step] / np.sum(X[seq_idx][time_step])
    # sort the array by descending order
    sorted_idx = np.argsort(-total_abundance)
    sorted_bacteria = [idx2bacteria[idx] for idx in sorted_idx]
    return sorted_idx, sorted_bacteria


'''
    Pick the dimensions of the bacteria that meets the requirement (top k or in includes, not in excludes)
    
    # Arguments
        idx2bacteria: a map from a dimension in the numpy array to its corresponding bacteria cluster
        sorted_idx: sorted (according to average relative abundance) index of the bacteria cluster
        k: number of top (in terms of relative abundance) clusters to pick, in addition to "include" argument
        includes: an array of bacteria clusters to be included (in addition to the top k clusters)
        excludes: an array of bacteria clusters to be excluded
    # Returns
        picked_dimension: a list of index corresponding to the top bacteria clusters
'''
def pick_dimension_list(idx2bacteria, sorted_idx, k, includes, excludes):
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


'''
    Create the returned X, given the picked dimensions
    
    # Arguments
        X: an array of (length=number of patients) array of (length=number of measurement span) of
            numpy array (shape=(number_of_bacteria_clusters,)) or None values (for missing) measurements
            include all clusters
        picked_dimensions: a list of index corresponding to the top bacteria clusters
        include_other: whether to include the 'OTHER' cluster in the returned X
    # Returns:
        returned_X: an array of (length=number of patients) array of (length=number of measurement span) of
            numpy array (shape=(number_of_bacteria_clusters,)) or None values (for missing) measurements
            include only the chosen clusters
'''
def create_returned_X(X, picked_dimensions, include_other):
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

'''
    Transform counts to frequencies
    
    # Arguments
        x: measurement
        epsilon: the smoothing factor. default 0 - equivalent to w\ out smoothing
    # Returns
        x / np.sum(x): the smoothed frequency of x
'''
def measurement_mean(x, epsilon=0):
    x += epsilon
    return x / np.sum(x)

'''
    SPIEC transform
    
    # Arguments
        x: measurement frequency (all entries add up to 1)
    # Returns
        y: spiec(x). y_i = log(x_{i} / geometric_mean(x)), except the last dimension;
            last dimension is dropped
'''
def SPIEC_transform(x):
    # ensure that the measurement is a frequency
    assert(np.abs(np.sum(x) - 1) < 1e-7)
    y = np.log(x)
    y -= np.mean(y)
    return y[:-1]

'''
    Inverse SPEIC transform
'''
def inverse_SPIEC(y):
    x = np.append(y, [1])
    x = np.exp(x)
    x = x / np.sum(x)
    return x

'''
    Default transformation: mean with smoothed factor=1 followed by SPIEC
'''
def default_transform(x):
    x = measurement_mean(x, epsilon=1)
    x = SPIEC_transform(x)
    return x

'''
    A wrapper that maps each measurement/control/events/arrays to an element,
    while perserving the missing measurements and the same data structure
    
    # Arguments
        X: an array of (length=number of patients) array of (length=number of measurement span) of
            numpy array (measurement/control/events/arrays) or None values (for missing) measurements
        f: the function applied to each numpy array
    # Returns
        returned_X: same with X, except that each measurement is applied f
'''
def map_2D_array(X, f):
    returned_X = [[f(x) if x is not None else None for x in x_seq] for x_seq in X]
    return returned_X

'''
    Apply default measurement transformation to X
'''
def default_measurement_transformation(X):
    return map_2D_array(X, default_transform)

'''
    Read the event table and parse it
    
    # Arguments
        file_name: the name of the event file
        id2start_date: a map from patientId (starting from 0) to the starting measurement date
            relative to the reference date
        id2end_date: a map from patientId (starting from 0) to the ending measurement date
            relative to the reference date
    # Returns
        U: an array of (length=number of patients) array of (length=number of measurement span) of
            numpy array (shape=(number_of_events,)), one hot encoded (1 if the corresponding dimension)
            occured
        event_name2idx: a map from (type, description) to dimension in each of the numpy array in U
        idx2event_name: a map from dimension in each of the numpy array in U to (type, description)
'''
def read_event(file_name, id2start_date, id2end_date):
    
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

'''
    Read a line of event description and add it to U
    
    # Arguments
        l: a line (string) of event description in the event file
        U: an array of (length=number of patients) array of (length=number of measurement span) of
            numpy array (shape=(number_of_events,)), one hot encoded (1 if the corresponding dimension)
            occured
        id2start_date: a map from patientId (starting from 0) to the starting measurement date
            relative to the reference date
        id2end_date: a map from patientId (starting from 0) to the ending measurement date
            relative to the reference date
        event_name2idx: a map from (type, description) to dimension in each of the numpy array in U
'''
def add_event(l, U, id2start_date, id2end_date, event_name2idx):
    try:
        id, event_name, start, end = parse_line_to_event(l)
        start_idx = max(0, start - id2start_date[id])
        end_idx = min(len(U[id]), end + 1 - id2start_date[id])
        for time_step in range(start_idx, end_idx):
            U[id][time_step][event_name2idx[event_name]] = 1
    except:
        return

'''
    initialize the returned U array
'''
def initialize_U(id2start_date, id2end_date, event_count):
    U = [np.zeros((id2end_date[id] - id2start_date[id] + 1, event_count))
         for id in id2start_date]
    return U

'''
    Read the event file and create a dict that maps the event name to an index
    
    # Arguments
        file_name: the event file name
    # Returns
        event_name2idx: a map from (type, description) to index
'''
def get_event_name2idx(file_name):
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

'''
    Parse a line of event description to a list of key elements of the event
    
    # Arguments
        line: a line (string) of event description in the event file
    # Returns
        id: patient id (starting from 0)
        event_name: the complete name of the event, (type, description)
            e.g. (Antibiotic, Fluoroquinolone)
        start: the start date of the event (inclusive) relative to the reference date
        end: the end date of the event (inclusive) relative to the reference date
'''
def parse_line_to_event(line):
    id, type, description, start, end = line.split('\t')
    id = int(id) - 1
    event_name = (type, description)
    start = int(start)
    end = int(end)
    return id, event_name, start, end

'''
    Pick the selected events of U according to the index they correspond
    
    # Arguments
        U: an array of (length=number of patients) array of (length=number of measurement span) of
            numpy array (shape=(number_of_events,)), one hot encoded (1 if the corresponding dimension)
            occured
        picked_dimensions: the selected dimensions of u in U
    
    # Returns
        returned_U: same with U, except that it only includes the selected dimensions of all u in U
'''
def pick_events_by_dims(U, picked_dimensions):
    returned_U = [[u[picked_dimensions] if u is not None else None for u in u_seq] for u_seq in U]
    return returned_U

'''
    Picking the events according to a filter function
    
    # Arguments:
        U: an array of (length=number of patients) array of (length=number of measurement span) of
            numpy array (shape=(number_of_events,)), one hot encoded (1 if the corresponding dimension)
            occured
        idx2event_name: a map from dimension in each of the numpy array in U to (type, description)
        filter: a function that takes in a complete event name, tuple (type, description) as argument
            returns True if the event is selected, False otherwise
    # Returns:
        returned_U: same with U, except that it only includes the selected dimensions of all u in U
        returned_idx2event_name: a map from dimension in each of the numpy array in U to (type, description) for returned_U
'''
def extract_event_by_filter(U, idx2event_name, filter):
    returned_idx2event_name, picked_dimensions = {}, []
    for idx in idx2event_name:
        event_name = idx2event_name[idx]
        if filter(event_name):
            returned_idx2event_name[len(returned_idx2event_name)] = event_name
            picked_dimensions.append(idx)
    returned_U = pick_events_by_dims(U, picked_dimensions)
    return returned_U, returned_idx2event_name

'''
    Picking the events that has the specified "type" (e.g. "Antibiotic", "Bacteremia", etc)
    
    # Arguments:
        U: an array of (length=number of patients) array of (length=number of measurement span) of
            numpy array (shape=(number_of_events,)), one hot encoded (1 if the corresponding dimension)
            occured
        idx2event_name: a map from dimension in each of the numpy array in U to (type, description)
        type: event type to be selected
    # Returns:
        returned_U: same with U, except that it only includes the selected dimensions of all u in U
        returned_idx2event_name: a map from dimension in each of the numpy array in U to (type, description) for returned_U
'''
def extract_event_by_type(U, idx2event_name, type):
    def filter(event_name):
        try:
            return event_name[0] == type
        except:
            return False
    return extract_event_by_filter(U, idx2event_name, filter)

'''
    For one patient, one time_step, if one entry is 1, it is considered 1
'''
def or_U(U):
    return map_2D_array(U, lambda x: x.any())
