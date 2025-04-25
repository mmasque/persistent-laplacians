def persistent_betti_numbers(persistence_data, unique_filtration_values):
    max_filt_index = len(unique_filtration_values) - 1
    persistent_bettis = {}
    for n in persistence_data.keys():
        bettis = persistence_data[n]
        persistent_bettis_n = {}
        for (i, j) in bettis:
            persistent_ij = (bettis.get((i, j-1), 0) - bettis.get((i, j), 0)) - (bettis.get((i-1, j-1), 0) - bettis.get((i-1, j), 0))
            if persistent_ij != 0 and i != j:
                persistent_bettis_n[(i, j)] = persistent_ij
        for i in range(len(unique_filtration_values)):
            persistent_i_inf = bettis.get((i, max_filt_index),0) - bettis.get((i-1, max_filt_index),0)
            if persistent_i_inf != 0:
                persistent_bettis_n[(i,-1)] = persistent_i_inf
        persistent_bettis[n] = persistent_bettis_n
    return persistent_bettis