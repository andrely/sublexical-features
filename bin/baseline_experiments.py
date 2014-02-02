from argparse import ArgumentParser
import logging
import os
import sys

cur_path, _ = os.path.split(__file__)
sys.path.append(os.path.join(cur_path, '..', 'Corpora'))
sys.path.append(os.path.join(cur_path, '..', 'Experiments'))
sys.path.append(os.path.join(cur_path, '..', 'BrownClustering'))

from experiments.experiment_runner import run_experiment, baseline_pipelines


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    argparser = ArgumentParser()
    argparser.add_argument('-i', '--experiment-id')
    argparser.add_argument('-p', '--processors', type=int)
    argparser.add_argument('-c', '--corpus-path')
    argparser.add_argument('-b', '--brown-cluster-path')
    args = argparser.parse_args()

    brown_cluster_path = os.getcwd()

    if args.brown_cluster_path:
        brown_cluster_path = args.brown_cluster_path

    logging.info("Using Brown clusters in %s" % brown_cluster_path)

    pipelines = baseline_pipelines(brown_cluster_path=brown_cluster_path)

    if args.experiment_id:
        experiments = args.experiment_id.split(',')
    else:
        experiments = pipelines.keys()

    logging.info("Running experiments %s" % ', '.join(experiments))

    n_jobs = 1

    if args.processors:
        n_jobs = args.processors

    logging.info("Using %d processors" % n_jobs)

    corpus_path = os.getcwd()

    if args.corpus_path:
        corpus_path = args.corpus_path

    logging.info("Using corpus at %s" % corpus_path)

    for exp_id in experiments:
        if not pipelines.has_key(exp_id):
            logging.warn("No experiment with id %s" % exp_id)
            continue

        logging.info("Running %d fold cross validation for experiment %s" % (10, exp_id))
        pipeline = pipelines[exp_id]
        score_mean, score_std, scores = run_experiment(corpus_path, pipeline, n_jobs=n_jobs)
        score_str = '\t'.join(["%.04f" % s for s in scores])
        sys.stdout.write("%s\t%.04f\t%.04f\t%s\n" % (exp_id, score_mean, score_std, score_str))
