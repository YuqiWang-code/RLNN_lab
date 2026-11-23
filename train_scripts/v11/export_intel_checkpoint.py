"""Export the 20 released INTEL GCN layers and a reference prediction from its TF graph."""
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf


def sparse_feed(matrix):
    matrix = sp.coo_matrix(matrix)
    return (np.column_stack((matrix.row, matrix.col)), matrix.data.astype(np.float32), matrix.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tf.disable_eager_execution()
    path = str(args.checkpoint)
    reader = tf.train.load_checkpoint(path)
    arrays = {}
    for layer in range(1, 21):
        for support in range(2):
            name = f"gcn_deep_diver/graphconvolution_{layer}_vars/weights_{support}"
            arrays[f"w{layer}_{support}"] = reader.get_tensor(name)
    rng = np.random.RandomState(17)
    upper = np.triu(rng.rand(64, 64) < 0.19, k=1)
    adjacency = np.asarray(upper | upper.T, dtype=np.float32)
    d = np.maximum(adjacency.sum(1), 1) ** -0.5
    identity = np.eye(len(d))
    supports = [identity, identity - d[:, None] * adjacency * d[None, :]]
    features = np.full((len(d), 32), 1 / 32)
    saver = tf.train.import_meta_graph(path + ".meta", clear_devices=True)
    graph = tf.get_default_graph()
    feed = {}
    for start, matrix in zip((0, 3, 6), [*supports, features]):
        # TF1 sparse_placeholder creates indices, values, shape. Match by dtype and rank.
        indices, values, shape = sparse_feed(matrix)
        for index in range(start, start + 3):
            name = "Placeholder" + (f"_{index}" if index else "") + ":0"
            tensor = graph.get_tensor_by_name(name)
            feed[tensor] = values if tensor.dtype == tf.float32 else (indices if tensor.shape.ndims == 2 else shape)
    feed[graph.get_tensor_by_name("Placeholder_11:0")] = [features.size]
    with tf.Session() as session:
        saver.restore(session, path)
        reference = session.run("concat_30:0", feed_dict=feed)
    np.savez(args.output, **arrays, fixture_adjacency=adjacency, fixture_maps=reference[:, 1::2])


if __name__ == "__main__":
    main()
