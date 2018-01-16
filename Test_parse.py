from Parse_Utils import read_otu_table, read_event

otu_file = '../data2/otuTableDI.tsv'
event_file = '../data2/eventsDI.tsv'
X, bacteria2idx, idx2bacteria, id2start_date, id2end_date, file_id2program_id, program_id2file_id \
    = read_otu_table(otu_file)
print(X[0][0])
U, event_name2idx, idx2event_name = read_event(event_file, id2start_date, id2end_date, file_id2program_id)
print(U[0][0])
print(event_name2idx)