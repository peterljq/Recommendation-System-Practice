# -*-coding:utf8-*-

from __future__ import division
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_feature_column():

    """
    age,workclass,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,label
    get wide feature and deep feature
    Return:
        wide feature column, deep feature column
    """
    print("Getting features.")
    # 连续特征
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education-num")
    capital_gain = tf.feature_column.numeric_column("capital-gain")
    capital_loss = tf.feature_column.numeric_column("capital-loss")
    hour_per_work = tf.feature_column.numeric_column("hours-per-week")

    # 离散特征哈希化，桶尺寸为品类最大数量
    work_class = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=512)
    education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=512)
    marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital-status", hash_bucket_size=512)
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=512)
    realationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=512)

    # 连续特征离散化，boundary为离散品类range
    age_bucket = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    gain_bucket = tf.feature_column.bucketized_column(capital_gain, boundaries=[0, 1000, 2000, 3000, 10000])
    loss_bucket = tf.feature_column.bucketized_column(capital_loss, boundaries=[0, 1000, 2000, 3000, 5000])

    # 离散化之后的特征交叉化，桶尺寸为品类最大数量
    cross_columns = [
        tf.feature_column.crossed_column([age_bucket, gain_bucket], hash_bucket_size=36),
        tf.feature_column.crossed_column([gain_bucket, loss_bucket], hash_bucket_size=16)
    ]

    base_columns = [work_class, education, marital_status, occupation, realationship, age_bucket, gain_bucket,
                    loss_bucket, ]

    # wide层放入：离散特征，连续特征的离散化，连续特征离散化后的交叉化
    wide_columns = base_columns + cross_columns

    # deep层放入：连续特征，离散特征的嵌入化（桶尺寸521 -> 维度9）
    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hour_per_work,
        tf.feature_column.embedding_column(work_class, 9),
        tf.feature_column.embedding_column(education, 9),
        tf.feature_column.embedding_column(marital_status, 9),
        tf.feature_column.embedding_column(occupation, 9),
        tf.feature_column.embedding_column(realationship, 9),
    ]
    return wide_columns, deep_columns


def build_model_estimator(wide_column, deep_column, model_folder):
    """
    Args:
        wide_column: wide feature
        deep_column:deep feature
        model_folder: origin model output folder
    Return:
        model_es, serving_input_fn
    """

    print("Model Building.")
    model_es = tf.estimator.DNNLinearCombinedClassifier(
        model_dir= model_folder,
        linear_feature_columns= wide_column,
        linear_optimizer=tf.train.FtrlOptimizer(0.1, l2_regularization_strength=1.0),
        dnn_feature_columns= deep_column,
        dnn_optimizer= tf.train.ProximalAdagradOptimizer(learning_rate=0.1, l1_regularization_strength=0.001,
                                                         l2_regularization_strength=0.001),
        dnn_hidden_units=[128, 64, 32, 16]
    )
    feature_column = wide_column + deep_column
    feature_spec = tf.feature_column.make_parse_example_spec(feature_column)
    serving_input_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    return model_es, serving_input_fn


def input_fn(data_file, re_time, shuffle, batch_num, predict):
    """
    Args:
        data_file:input data , train_data, test_data
        re_time:time to repeat the data file
        shuffle: shuffle or not [true or false]
        batch_num:
        predict: train or test [true or false]
    Return:
        train_feature, train_label or test_feature
    """
    _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

    _CSV_COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'label'
     ]

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
        classes = tf.equal(labels, '>50K')
        return features, classes

    def parse_csv_predict(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
        return features

    data_set = tf.data.TextLineDataset(data_file).skip(1).filter(lambda line: tf.not_equal(tf.strings.regex_full_match(line, ".*\?.*"),True))
    if shuffle:
        data_set = data_set.shuffle(buffer_size=30000)
    if predict:
        data_set = data_set.map(parse_csv_predict, num_parallel_calls=5)
    else:
        data_set = data_set.map(parse_csv, num_parallel_calls=5)

    data_set = data_set.repeat(re_time)
    data_set = data_set.batch(batch_num)
    return data_set


def train_wd_model(model_es, train_file, test_file, model_export_folder, serving_input_fn):
    """
    Args:
        model_es: wd estimator
        train_file:
        test_file:
        model_export_folder: model export for tf serving
        serving_input_fn: function for model export
    """
    print("Training Model.")

    model_es.train(input_fn=lambda: input_fn(train_file, 10, True, 100, False))
    print(model_es.evaluate(input_fn=lambda:input_fn(test_file, 1, False, 100, False)))
    model_es.export_savedmodel(model_export_folder, serving_input_fn)




def run_main(train_file, test_file, model_folder, model_export_folder):
    wide_column, deep_column = get_feature_column()
    model_es, serving_input_fn=build_model_estimator(wide_column, deep_column, model_folder)
    train_wd_model(model_es, train_file, test_file, model_export_folder, serving_input_fn)

if __name__ == '__main__':
    run_main("../data/train.txt", "../data/test.txt", "../data/wd", "../data/wd_export")