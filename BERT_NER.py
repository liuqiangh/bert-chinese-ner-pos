#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:zhoukaiyin
Adjust code for chinese ner
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from lstm_crf_layer import BLSTM_CRF
from tensorflow.python.ops import math_ops
import tf_metrics
import pickle
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

# Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False,
                  "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 40.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("epochs_per_eval", 300,
                     "How many steps to train in every epoch")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_float("dropout_rate", 0.5,
                   "Dropout rate")

flags.DEFINE_integer("lstm_size", 128,
                     "size of lstm units.")

flags.DEFINE_string("cell", "lstm",
                    "which rnn cell used.")

flags.DEFINE_integer("num_layers", 1,
                     "number of rnn layers, default is 1.")
flags.DEFINE_integer("save_summary_steps", 500,
                     "save_summary_steps.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with open(input_file, encoding="UTF-8") as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.rstrip()
                word = line.rstrip().split('\t')[0]
                label = line.rstrip().split('\t')[-1]
                # word = line.strip().split(' ')[0]
                # label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                # if len(contends) == 0 and words[-1] == '。':
                if len(contends) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                # 去掉原语料中无意义的空格
                if word != ' ' and len(word) > 0:
                    words.append(word)
                    labels.append(label)
            return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "test")

    def get_labels(self):
        # prevent potential bug for chinese text mixed with english text
        # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]","[SEP]"]
        # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        # clinical data-msra
        return ["O", "B-TREATMENT", "I-TREATMENT", "B-BODY", "I-BODY",
                "B-SIGNS", "I-SIGNS", "B-CHECK", "I-CHECK",
                "B-DISEASE", "I-DISEASE", "X", "[CLS]", "[SEP]"]
        # clinical data-rujin
        # return ["O", "B-Amount", "I-Amount",
        #         "B-Anatomy", "I-Anatomy", "B-Disease", "I-Disease",
        #         "B-Drug", "I-Drug", "B-Duration", "I-Duration",
        #         "B-Frequency", "I-Frequency", "B-Level", "I-Level",
        #         "B-Method", "I-Method", "B-Operation", "I-Operation",
        #         "B-Reason", "I-Reason", "B-SideEff", "I-SideEff",
        #         "B-Symptom", "I-Symptom", "B-Test", "I-Test",
        #         "B-Test_Value", "I-Test_Value", "B-Treatment", "I-Treatment",
        #         "X", "[CLS]", "[SEP]"]
        # clinical data-ruijin/5 classes
        # return ["O", "B-Anatomy", "I-Anatomy", "B-Disease", "I-Disease",
        #         "B-Symptom", "I-Symptom", "B-Test", "I-Test",
        #         "B-Treatment", "I-Treatment",
        #         "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def write_tokens(tokens, label_ids, mode):
    if mode == "test":
        file_name = FLAGS.output_dir.split('/')[1]
        with open('./' + file_name + '/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")
        wf = open(path, 'a', encoding='utf-8')
        for i, token in enumerate(tokens):
            if token != "**NULL**":
                line = token + ' ' + id2label[label_ids[i]] + '\n'
                wf.write(line)
        wf.close()


def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, mode):
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    assert len(textlist) == len(labellist)
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        # print(token)
        tokens.extend(token)
        label_1 = labellist[i]
        # print(label_1)
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
            # 此处是针对一些较长的英文词进行再切割，切割后第一个字符为原label，
            # 剩余的label均为'X'
        # print(tokens, labels)
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" %
                        " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    write_tokens(ntokens, label_ids, mode)
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None
):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    file_name = FLAGS.output_dir.split('/')[1]
    with open('./' + file_name + '/label2id.pkl', 'wb') as w:
        pickle.dump(label_map, w)

    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))
        feature = convert_single_example(
            ex_index, example, label_map, max_seq_length, tokenizer, mode)

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def filed_based_convert_examples_to_features_pre(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None
):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    file_name = FLAGS.output_dir.split('/')[1]
    with open('./' + file_name + '/label2id.pkl', 'wb') as w:
        pickle.dump(label_map, w)
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))
        feature = convert_single_example(
            ex_index, example, label_map, max_seq_length, tokenizer, mode)
        features.append(feature)

    return features


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        # 解码函数
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        # 将features解码
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def raw_serving_input_fn():
    input_ids = tf.placeholder(
        tf.int64, shape=[None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(
        tf.int64, shape=[None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(
        tf.int64, shape=[None, FLAGS.max_seq_length], name='segment_ids')
    label_ids = tf.placeholder(
        tf.int64, shape=[None, FLAGS.max_seq_length], name='label_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'label_ids': label_ids,
    })()
    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings,
                 dropout_rate=1.0, lstm_size=1, cell='lstm', num_layers=1):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    bert_layer = model.get_sequence_output()
    a = bert_layer.get_shape()
    max_seq_length = bert_layer.shape[1].value
    # 算序列真实长度
    used = tf.sign(tf.abs(input_ids))
    # [batch_size] 大小的向量，包含了当前batch中的序列长度
    lengths = tf.reduce_sum(used, reduction_indices=1)
    # 这里添加pos embedding层
    # 添加CRF output layer
    blstm_crf = BLSTM_CRF(embedded_chars=bert_layer, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers,
                          dropout_rate=dropout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer(crf_only=True)

    return rst

    # output_weight = tf.get_variable(
    #     "output_weights", [num_labels, hidden_size],
    #     initializer=tf.truncated_normal_initializer(stddev=0.02)
    # )
    # output_bias = tf.get_variable(
    #     "output_bias", [num_labels], initializer=tf.zeros_initializer()
    # )
    # with tf.variable_scope("loss"):
    #     if is_training:
    #         output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    #     output_layer = tf.reshape(output_layer, [-1, hidden_size])
    #     logits = tf.matmul(output_layer, output_weight, transpose_b=True)
    #     logits = tf.nn.bias_add(logits, output_bias)
    #     logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, 11])
    # mask = tf.cast(input_mask,tf.float32)
    # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
    # return (loss, logits, predict)
    ##########################################################################
    # log_probs = tf.nn.log_softmax(logits, axis=-1)
    # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    # loss = tf.reduce_sum(per_example_loss)
    # probabilities = tf.nn.softmax(logits, axis=-1)
    # predict = tf.argmax(probabilities,axis=-1)
    # return (loss, per_example_loss, logits,predict)
    ##########################################################################


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # batch数据导入
        (total_loss, logits, trans, pred_ids) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings, FLAGS.dropout_rate, FLAGS.lstm_size, FLAGS.cell, FLAGS.num_layers)
        tvars = tf.trainable_variables()
        # 加载BERT模型
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            # if use_tpu:
            #     def tpu_scaffold():
            #         tf.train.init_from_checkpoint(
            #             init_checkpoint, assignment_map)
            #         return tf.train.Scaffold()
            #
            #     scaffold_fn = tpu_scaffold
            # else:
            #     tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            tf.summary.scalar('loss', total_loss)
            # 针对NER有修改
            hook_dict = {}
            hook_dict['loss'] = total_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=FLAGS.save_summary_steps)

            tf.estimator.Estimator
            tf.estimator.train_and_evaluate
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])
            # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            #     mode=mode,
            #     loss=total_loss,
            #     train_op=train_op,
            #     scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            # 针对NER ,进行了修改
            # def metric_fn(label_ids, pred_ids):
            #     return {
            #         "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),
            #     }

            # eval_metrics = metric_fn(label_ids, pred_ids)
            # output_spec = tf.estimator.EstimatorSpec(
            #     mode=mode,
            #     loss=total_loss,
            #     eval_metric_ops=eval_metrics
            # )
            # hook_dict = {}

            def metric_fn(label_ids, pred_ids, num_labels):
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                pos_indices = [id for id in range(2, num_labels - 3)]
                # pos_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                #                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                # pos_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                precision = tf_metrics.precision(
                    label_ids, pred_ids, num_labels, pos_indices, average="micro")
                recall = tf_metrics.recall(
                    label_ids, pred_ids, num_labels, pos_indices, average="micro")
                f = tf_metrics.f1(label_ids, pred_ids,
                                  num_labels, pos_indices, average="micro")
                # hook_dict['precision'] = precision
                # hook_dict['recall'] = recall
                # hook_dict['f'] = f
                # tf.summary.scalar('precision', precision)
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            # eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            eval_metrics = (
                metric_fn, [label_ids, pred_ids, num_labels])

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ner": NerProcessor
    }
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    # max_position_embeddings设置为512，可增大序列长度
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    # if FLAGS.use_tpu and FLAGS.tpu_name:
    #     tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    #         FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    # tf.contrib.tpu.RunConfig
    # tf.contrib.estimator
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    eval_examples = None
    eval_file = None
    eval_best_f = 0

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        # 控制初始学习率
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        # for i in range(num_train_steps // FLAGS.epochs_per_eval):
        # version1.0
        # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        # estimator.train(input_fn=train_input_fn,
        #                 steps=FLAGS.epochs_per_eval)
        #
        # tf.logging.info("***** Running evaluation *****")
        # tf.logging.info("  Num examples = %d", len(eval_examples))
        # tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        # eval_steps = None
        # # if FLAGS.use_tpu:
        # #     eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        # # eval_drop_remainder = True if FLAGS.use_tpu else False
        # eval_drop_remainder = False
        # eval_input_fn = file_based_input_fn_builder(
        #     input_file=eval_file,
        #     seq_length=FLAGS.max_seq_length,
        #     is_training=False,
        #     drop_remainder=eval_drop_remainder)
        # result = estimator.evaluate(
        #     input_fn=eval_input_fn, steps=eval_steps)
        # f1_score = result['eval_f']
        # if f1_score > eval_best_f:
        #     output_eval_file = os.path.join(
        #         FLAGS.output_dir, "eval_results.txt")
        #     with open(output_eval_file, "w") as writer:
        #         tf.logging.info("***** Eval results *****")
        #         for key in sorted(result.keys()):
        #             tf.logging.info("  %s = %s", key, str(result[key]))
        #             writer.write("%s = %s\n" % (key, str(result[key])))
        #     eval_best_f = f1_score
        #     estimator.export_saved_model(
        #         FLAGS.output_dir + '/eval', raw_serving_input_fn)
        # version2.0
        eval_drop_remainder = False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                            max_steps=num_train_steps)
        exporter = tf.estimator.BestExporter(
            serving_input_receiver_fn=raw_serving_input_fn,
            exports_to_keep=2)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                          steps=None,  # steps=None, evaluate on the entire eval dataset
                                          exporters=exporter)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if not FLAGS.do_train and FLAGS.do_eval:

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=FLAGS.output_dir + 'model.ckpt-41000')
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results_s.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        file_name = FLAGS.output_dir.split('/')[1]
        with open('./' + file_name + '/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        features = filed_based_convert_examples_to_features_pre(predict_examples, label_list,
                                                                FLAGS.max_seq_length, tokenizer,
                                                                predict_file, mode="test")
        export_dir = FLAGS.output_dir + 'eval/1594610385'
        predict_fn = tf.contrib.predictor.from_saved_model(export_dir)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")

        predictions = []
        for feature in features:
            result = predict_fn({
                "input_ids": [feature.input_ids],
                "input_mask": [feature.input_mask],
                "segment_ids": [feature.segment_ids],
            })
            prediction = np.squeeze(result['output']).tolist()
            predictions.extend(prediction)
        with open(output_predict_file, 'w') as writer:
            output_line = "\n".join(
                id2label[id] for id in predictions if id != 0) + "\n"
            writer.write(output_line)
            writer.close()
        # 将token、真实label和预测label写入文件
        token_file = os.path.join(FLAGS.output_dir, "token_test.txt")
        original_predict_file = os.path.join(
            FLAGS.output_dir, "original_predict_test.txt")
        f1 = open(original_predict_file, "w+", encoding="utf-8")
        f2 = open(token_file, encoding="utf-8")
        f3 = open(output_predict_file, encoding="utf-8")
        list_f2 = list(f2)
        list_f3 = list(f3)
        for i, line in enumerate(list_f2):
            f2_line = line.strip()
            if f2_line == '[SEP] [SEP]':
                pass
            elif f2_line == '[CLS] [CLS]':
                pass
            else:
                f3_line = list_f3[i].strip()
                newline = f2_line + ' ' + f3_line + '\n'
                f1.write(newline)

        # tf.logging.info("***** Running prediction*****")
        # tf.logging.info("  Num examples = %d", len(predict_examples))
        # tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        # if FLAGS.use_tpu:
        #     # Warning: According to tpu_estimator.py Prediction on TPU is an
        #     # experimental feature and hence not supported here
        #     raise ValueError("Prediction in TPU not supported")
        # predict_drop_remainder = True if FLAGS.use_tpu else False
        # predict_input_fn = file_based_input_fn_builder(
        #     input_file=predict_file,
        #     seq_length=FLAGS.max_seq_length,
        #     is_training=False,
        #     drop_remainder=predict_drop_remainder)
        #
        # result = estimator.predict(
        #     input_fn=predict_input_fn, checkpoint_path=FLAGS.output_dir + 'model.ckpt-3000')
        # output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        # # 将预测结果写入文件
        # with open(output_predict_file, 'w') as writer:
        #     for prediction in result:
        #         output_line = "\n".join(
        #             id2label[id] for id in prediction if id != 0) + "\n"
        #         writer.write(output_line)
        #     writer.close()
        # # 将token、真实label和预测label写入文件
        # token_file = os.path.join(FLAGS.output_dir, "token_test.txt")
        # original_predict_file = os.path.join(
        #     FLAGS.output_dir, "original_predict_test.txt")
        # f1 = open(original_predict_file, "w+", encoding="utf-8")
        # f2 = open(token_file, encoding="utf-8")
        # f3 = open(output_predict_file, encoding="utf-8")
        # list_f2 = list(f2)
        # list_f3 = list(f3)
        # for i, line in enumerate(list_f2):
        #     f2_line = line.strip()
        #     if f2_line == '[SEP] [SEP]':
        #         pass
        #     elif f2_line == '[CLS] [CLS]':
        #         pass
        #     else:
        #         f3_line = list_f3[i].strip()
        #         newline = f2_line + ' ' + f3_line + '\n'
        #         f1.write(newline)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
