import tensorflow as tf  # TF 1.1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import shutil

tf.logging.set_verbosity(tf.logging.INFO)
np.random.seed(1)
tf.set_random_seed(1)
from sklearn.metrics import roc_auc_score
# ==============================================================================
# ==============================================================================
# Read in data
df = pd.read_csv('temp.tsv', sep='\t', nrows=5e5)


del df['temp.id']


df_train, df_test = train_test_split(df, test_size=0.2)

LABEL_COLUMN = 'temp.target'

# remove NaN elements
df_train = df_train.dropna(how='any', axis=0)
df_test = df_test.dropna(how='any', axis=0)


CATEGORICAL_COLUMNS = [key for key in dict(
    df.dtypes) if dict(df.dtypes)[key] in ['O']]

CONTINUOUS_COLUMNS = [key for key in dict(
    df.dtypes) if dict(df.dtypes)[key] not in ['O']]

CONTINUOUS_COLUMNS.remove(LABEL_COLUMN)

df[CONTINUOUS_COLUMNS].apply(lambda x: (x - np.mean(x)) / np.std(x))

# ==============================================================================
# ==============================================================================


def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1])
                       for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(list(continuous_cols.items()) +
                        list(categorical_cols.items()))

    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def build_estimator(model_dir, model_type, early_stopping_parameters,  wide_opt,
                    deep_opt, wide_n_deep_opt, sparse_interactions):
    """Build an estimator."""
    # Sparse base columns.

    categorical_sparse_hashed = list(tf.contrib.layers.sparse_column_with_hash_bucket(k,
                                                                                      combiner='sqrtn', hash_bucket_size=1e3)
                                     for k in CATEGORICAL_COLUMNS)
    # Continuous base columns.

    continous = [tf.contrib.layers.real_valued_column(
        k) for k in CONTINUOUS_COLUMNS]

    if sparse_interactions == True:
        sparse_combinations = list(
            itertools.combinations(categorical_sparse_hashed, 2))
        sparse_interactions = [tf.contrib.layers.crossed_column([k1, k2],
                                                                hash_bucket_size=int(
                                                                    1e4),
                                                                hash_key=0xDECAFCAFFE, combiner='sqrtn')
                               for k1, k2 in sparse_combinations]
    else:
        sparse_interactions = []


# Wide columns and deep columns.
wide_columns = continous + categorical_sparse_hashed + sparse_interactions


if model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns,
                                          optimizer=wide_opt['optimizer'],
                                          config=tf.contrib.learn.RunConfig(
                                              save_checkpoints_steps=early_stopping_parameters[
                                                  'save_checkpoints_steps'],
                                              save_checkpoints_secs=None))

elif model_type == "deep":
    deep_columns = continous + list(tf.contrib.layers.embedding_column(k,
                                                                       dimension=deep_opt['embedding_dimensions']) for k in
                                    categorical_sparse_hashed)
m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                   feature_columns=deep_columns,
                                   hidden_units=deep_opt['hidden_units'],
                                   optimizer=deep_opt['optimizer'],
                                   config=tf.contrib.learn.RunConfig(
                                       save_checkpoints_steps=early_stopping_parameters[
                                           'save_checkpoints_steps'],
                                       save_checkpoints_secs=None))

else:
    deep_columns = list(tf.contrib.layers.embedding_column(k,
                                                           dimension=wide_n_deep_opt['embedding_dimensions']) for k in
                        categorical_sparse_hashed) + continous
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    linear_optimizer=wide_n_deep_opt['linear_optimizer'],
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=wide_n_deep_opt['hidden_units'],
    dnn_optimizer=wide_n_deep_opt['dnn_optimizer'],
    config=tf.contrib.learn.RunConfig(
        save_checkpoints_steps=early_stopping_parameters['save_checkpoints_steps'],
        save_checkpoints_secs=None),
    fix_global_step_increment_bug=Ture)
return m


def train_and_eval(model_dir, model_type, train_steps,
                   early_stopping_parameters,
                   wide_opt, deep_opt, wide_n_deep_opt, sparse_interactions=False):
    try:
    shutil.rmtree(model_dir)


except:
    print("model_dir does not exist.")

""
"Train and evaluate the model."
""
print("model directory = %s" % model_dir)

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=lambda: input_fn(df_test),
    every_n_steps=early_stopping_parameters['every_n_steps'],
    eval_steps=1,
    early_stopping_metric="auc",
    early_stopping_metric_minimize=False,
    early_stopping_rounds=early_stopping_parameters[
        'early_stopping_rounds'])

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        tf.get_default_graph()
        tf.set_random_seed(1)

        m = build_estimator(model_dir, model_type, early_stopping_parameters,
                            wide_opt, deep_opt, wide_n_deep_opt, sparse_interactions)

        m.fit(input_fn=lambda: input_fn(df_train),
              steps=train_steps, monitors=[validation_monitor])

        tf.logging.set_verbosity(tf.logging.ERROR)

        results = m.evaluate(input_fn=lambda: input_fn(df_train), steps=1)
        print("The evaluation results on train data are: \n")
        for key in sorted(results):
            print("%s: %s" % (key, results[key]))

        results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
        print("The evaluation results on test data are: \n")
        for key in sorted(results):
            print("%s: %s" % (key, results[key]))

        predict = list(m.predict_proba(input_fn=lambda: input_fn(df_test)))
        predict_one = [k[1] for k in predict]
        print("sklearn test auc %s:" % roc_auc_score(
            df_test[LABEL_COLUMN], predict_one))

        auc = tf.metrics.auc(labels=np.array(df_test[LABEL_COLUMN]),
                             predictions=np.array(predict_one))
        sess.run(tf.local_variables_initializer())
        print("tensorflow test auc:", list(sess.run(auc))[1])
        sess.close()

if __name__ == "__main__":
    model_type = 'deep'  # 'deep' #  'wide_n_deep'
    model_dir = './tf1_' + model_type
#    model_dir = None
    train_steps = 1e6
    early_stopping_parameters = {'every_n_steps': 100, 'save_checkpoints_steps': 50,
                                 'early_stopping_rounds': 1000}

    wide_opt = {'optimizer': None}
#        tf.train.FtrlOptimizer(
#            learning_rate = 0.1,
#            l1_regularization_strength=0.0,
#            l2_regularization_strength=0.1)}

    deep_opt = {'embedding_dimensions': 10, 'hidden_units': [100, 50, 25],
                'optimizer': tf.train.AdadeltaOptimizer(learning_rate=0.3), 'dropout': 0.25}
    # tf.train.AdamOptimizer(learning_rate=0.1)}
    wide_n_deep_opt = {'embedding_dimensions': 10, 'hidden_units': [100, 50],
                       'linear_optimizer': None,
                       # tf.train.AdadeltaOptimizer(learning_rate=0.1),
                       'dnn_optimizer': None}
#           tf.train.ProximalAdagradOptimizer(
#                learning_rate=0.05,
#                l1_regularization_strength=0.0,
#                l2_regularization_strength=0.01)}

    train_and_eval(model_dir, model_type, train_steps,
                   early_stopping_parameters, wide_opt, deep_opt,
                   wide_n_deep_opt, sparse_interactions=False)


# ==============================================================================
# ==============================================================================
# Code change for SDCAoptimizer

# Added in function input_fn
    #example_col = {k: tf.constant(df[k].values) for k in [EXAMPLE_ID_COLUMN]}
    # USED IN SDCA optimizer
#  feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()) + list(example_col.items()))


# wide_opt = {'optimizer': tf.contrib.linear_optimizer.SDCAOptimizer(example_id_column = EXAMPLE_ID_COLUMN,
# symmetric_l1_regularization=0.0, symmetric_l2_regularization=0.1)}


# ==============================================================================
# ==============================================================================
# Streaming_auc can be used in validation monitor, we have train_auc, validate_auc and streaming_auc computed
#  validation_metrics = {
#    "validate_auc":
#        tf.contrib.learn.MetricSpec(
#            metric_fn=tf.metrics.auc,
#            prediction_key=tf.contrib.learn.PredictionKey.
#            CLASSES),
#
#    "streaming_auc":
#        tf.contrib.learn.MetricSpec(
#            metric_fn=tf.contrib.metrics.streaming_auc,
#            prediction_key=tf.contrib.learn.PredictionKey.
#            CLASSES)}


#    init_op = tf.global_variables_initializer()
#
#    with tf.Session() as sess:
#      sess.run(init_op)
#    tf.reset_default_graph()
