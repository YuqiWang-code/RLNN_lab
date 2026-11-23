"""Export the locally trained DIMES-MIS weights and an official inference fixture."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
import tensorflow.compat.v1 as tf

from train_dimes import OUT, WORK

sys.path.insert(0, str(WORK / "upstream/gcn"))
from models import GCN_DEEP_DIVER_DIMES
from reinforce import np_get_mis_solution
from utils import construct_feed_dict4pred, preprocess_features, simple_polynomials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=OUT / "dimes_er_small.npz")
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    tf.disable_v2_behavior()
    tf.flags.DEFINE_integer("hidden1", 64, "")
    tf.flags.DEFINE_integer("diver_num", 1, "")
    tf.flags.DEFINE_integer("num_layer", 20, "")
    tf.flags.DEFINE_float("weight_decay", 5e-4, "")
    tf.flags.DEFINE_float("learning_rate", 0.001, "")
    placeholders = dict(
        support=[tf.sparse_placeholder(tf.float32) for _ in range(2)],
        features=tf.sparse_placeholder(tf.float32, shape=(None, 32)),
        labels=tf.placeholder(tf.float32, shape=(None, 2)),
        labels_mask=tf.placeholder(tf.int32), max_graph_size=tf.placeholder(tf.int64),
        max_num_edges=tf.placeholder(tf.int64),
        dropout=tf.placeholder_with_default(0., shape=()),
        num_features_nonzero=tf.placeholder(tf.int32))
    model = GCN_DEEP_DIVER_DIMES(placeholders, input_dim=32, logging=False,
                                normalize_factor=1., meta_update=False, meta_steps=1)
    checkpoint = str(args.checkpoint) if args.checkpoint else tf.train.latest_checkpoint(str(OUT))
    if not checkpoint or (not args.checkpoint and not all((OUT / f"{epoch:04d}/score.txt").is_file() for epoch in range(4))):
        raise ValueError(f"Incomplete four-epoch DIMES retraining in {OUT}")
    with tf.Session() as session:
        saver = tf.train.Checkpoint(optimizer=model.optimizer, variables=tf.trainable_variables())
        status = saver.restore(checkpoint)
        status.assert_existing_objects_matched()
        status.run_restore_ops(session=session)
        arrays = {}
        graph_layer = 0
        for layer in model.weights:
            if "weights_0" in layer:
                graph_layer += 1
                for i in (0, 1):
                    arrays[f"w{graph_layer}_{i}"] = session.run(layer[f"weights_{i}"])
            else:
                arrays.update({f"dense_{name}": session.run(var) for name, var in layer.items()})
        rng = np.random.default_rng(32872)
        n = 47
        upper = np.triu(rng.random((n, n)) < .17, k=1)
        np.fill_diagonal(upper[:, 1:], True)
        adj = (upper | upper.T).astype(np.float32)
        looped = sparse.csr_matrix(adj + np.eye(n, dtype=np.float32))
        features = preprocess_features(sparse.lil_matrix(np.ones((n, 32), np.float32)))
        support = simple_polynomials(looped, 1)
        feed = construct_feed_dict4pred(features, support, placeholders)
        scores = session.run(model.outputs_softmax, feed_dict=feed)
        logits = scores[:, 1::2] - scores[:, 0::2]
        edges = np.stack(looped.nonzero(), axis=1)
        candidate = np_get_mis_solution(edges, logits, 1, n, len(edges))[0]
        arrays.update(fixture_adjacency=adj, fixture_logits=logits,
                      fixture_greedy_nodes=np.flatnonzero(candidate))
    if not all(np.isfinite(val).all() for val in arrays.values()):
        raise ValueError("Non-finite fitted weights or score fixture")
    target = args.output
    np.savez_compressed(target, **arrays)
    print(f"{graph_layer} GCN layers, {int(candidate.sum())} fixture vertices -> {target}")


if __name__ == "__main__":
    main()
