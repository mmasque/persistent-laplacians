import numpy as np
def compute_barcodes(persistence_data, unique_filtration_values):
    max_filt_index = len(unique_filtration_values) - 1
    barcodes = {}
    for n in persistence_data.keys():
        bettis = persistence_data[n]
        barcodes_n = {}
        pairs = [(i, j) for i in range(len(unique_filtration_values)) for j in range(i, len(unique_filtration_values))]
        for (i, j) in pairs:
            persistent_ij = (bettis.get((i, j-1), 0) - bettis.get((i, j), 0)) - (bettis.get((i-1, j-1), 0) - bettis.get((i-1, j), 0))
            if persistent_ij != 0 and i != j:
                barcodes_n[(i, j)] = persistent_ij
        for i in range(len(unique_filtration_values)):
            persistent_i_inf = bettis.get((i, max_filt_index),0) - bettis.get((i-1, max_filt_index),0)
            if persistent_i_inf != 0:
                barcodes_n[(i,np.inf)] = persistent_i_inf
        barcodes[n] = barcodes_n
    return barcodes