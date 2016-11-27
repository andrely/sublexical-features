import datetime
import os
import time

import tensorflow as tf
from gensim.corpora import Dictionary
from numpy import vstack, array
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

from sublexical_semantics.data.preprocessing import pad_sentences, clean_str, batch_iter
from sublexical_semantics.data.sentence_polarity import sentence_polarity_dataframe


def main():
    data = sentence_polarity_dataframe('../../data/sentence-polarity/')

    sentences = pad_sentences([clean_str(s).split() for s in data.sentence.values])
    vocab = Dictionary(sentences)

    y = LabelEncoder().fit_transform(data.polarity.values)
    y = vstack([y, 1 - y]).T

    x = [array([vocab.token2id[tok] for tok in sent]) for sent in sentences]

    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=.2)

    seq_length = len(x[0])
    num_filters = 128
    filter_sizes = [3, 4, 5]
    emb_size = 128

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            input_x = tf.placeholder(tf.int32, [None, seq_length], name='input_x')
            input_y = tf.placeholder(tf.float32, [None, 2], name='input_y')
            dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

            with tf.device('/cpu:0'), tf.name_scope('embedding'):
                W = tf.Variable(tf.random_uniform([len(vocab), emb_size], -1.0, 1.0), name='W')
                embedded_chars = tf.nn.embedding_lookup(W, input_x)
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

            pooled_outputs = []

            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % filter_size):
                    filter_shape = [filter_size, emb_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=.1), name='W')
                    b = tf.Variable(tf.constant(.1, shape=[num_filters]), name='b')
                    conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1],
                                        padding='VALID', name='conv')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    pooled = tf.nn.max_pool(h, ksize=[1, seq_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID', name='pool')
                    pooled_outputs.append(pooled)

            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(3, pooled_outputs)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            with tf.name_scope('dropout'):
                h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

            with tf.name_scope('output'):
                W = tf.Variable(tf.truncated_normal([num_filters_total, 2], stddev=.1), name='W')
                b = tf.Variable(tf.constant(.1, shape=[2]), name='b')
                scores = tf.nn.xw_plus_b(h_drop, W, b, name='scores')
                predictions = tf.argmax(scores, 1, name='predictions')

            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(scores, input_y)
                loss = tf.reduce_mean(losses)

            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
            print('Writing to {}\n'.format(out_dir))
            loss_summary = tf.scalar_summary('loss', loss)
            acc_summary = tf.scalar_summary('accuracy', accuracy)
            train_summary_op = tf.merge_summary([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.all_variables())

            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    input_x: x_batch,
                    input_y: y_batch,
                    dropout_keep_prob: .5
                }
                _, step, summaries, l, a = sess.run(
                    [train_op, global_step, train_summary_op, loss, accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, l, a))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                feed_dict = {
                    input_x: x_batch,
                    input_y: y_batch,
                    dropout_keep_prob: 1.0
                }
                step, summaries, l, a = sess.run(
                    [global_step, dev_summary_op, loss, accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, l, a))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = batch_iter(
                zip(x_train, y_train), 64, 200)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 100 == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % 100 == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == '__main__':
    main()