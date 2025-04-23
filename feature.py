
import numpy as np

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}

def seq_to_features_with_properties(sequence, max_seq_length):
    features = np.zeros(len(amino_acids))
    for aa in sequence:
        if aa in aa_to_int:
            features[aa_to_int[aa]] += 1
    return features / len(sequence)
