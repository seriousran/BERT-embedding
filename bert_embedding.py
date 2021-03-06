import os
import sys
import collections
import math
import numpy as np
import tensorflow as tf
from bert import modeling, tokenization
from bert.extract_features import input_fn_builder, model_fn_builder, convert_examples_to_features, InputExample

tf.logging.set_verbosity(tf.logging.ERROR)

curr_path = os.path.dirname(os.path.realpath(__file__))

# https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
DIR = curr_path + '/bert/'
MODEL_DIR  = DIR + 'multi_cased_L-12_H-768_A-12/'
CONFIG = MODEL_DIR + 'bert_config.json'
CKPT  = MODEL_DIR + 'bert_model.ckpt'
VOCAB  = MODEL_DIR + 'vocab.txt'

_max_seq_length = 512
_layers = [-1, -2, -3, -4]

class BERT(object):
  def init(self):
    bert_config = modeling.BertConfig.from_json_file(CONFIG)
    model_fn = model_fn_builder(bert_config=bert_config,
                                                  init_checkpoint=CKPT,
                                                  layer_indexes=_layers,
                                                  use_tpu=False,
                                                  use_one_hot_embeddings=False)
    self._estimator = tf.contrib.tpu.TPUEstimator(model_fn=model_fn,
                                            model_dir=MODEL_DIR,
                                            use_tpu=False,
                                            predict_batch_size=32,
                                            config=tf.contrib.tpu.RunConfig(master=None, tpu_config=tf.contrib.tpu.TPUConfig(num_shards=8, per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2)))
    self._tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB, do_lower_case=False)

  def extract_v1(self, sentence):
    
    example = [InputExample(unique_id=0, text_a=sentence, text_b=None)]
    features = convert_examples_to_features(examples=example,
                                                  seq_length=_max_seq_length,
                                                  tokenizer=self._tokenizer)
    input_fn = input_fn_builder(features=features,
                                                  seq_length=_max_seq_length)
    outputs = []
    for output in self._estimator.predict(input_fn):
      outputs.append(output)
    
    return outputs[0]
    
  
  def extracts_v1(self, sentences):
    
    examples = []
    for idx, sentence in enumerate(sentences):
      examples.append(InputExample(unique_id=idx, text_a=sentence, text_b=None))
    features = convert_examples_to_features(examples=examples,
                                                  seq_length=_max_seq_length,
                                                  tokenizer=self._tokenizer)
    input_fn = input_fn_builder(features=features,
                                                  seq_length=_max_seq_length)
    outputs = []
    for output in self._estimator.predict(input_fn):
      outputs.append(output)
    
    return outputs
  
  def extract(self, sentence):
    
    example = [InputExample(unique_id=0, text_a=sentence, text_b=None)]
    features = convert_examples_to_features(examples=example,
                                                  seq_length=_max_seq_length,
                                                  tokenizer=self._tokenizer)
    input_fn = input_fn_builder(features=features,
                                                  seq_length=_max_seq_length)
    outputs = []
    for output in self._estimator.predict(input_fn):
      feature = features[0]
      output_dict = collections.OrderedDict()
      all_features = []
      for (i, token) in enumerate(feature.tokens):
        all_layers = []
        for (j, layer_index) in enumerate(_layers):
          layer_output = output["layer_output_%d" % j]
          layers = collections.OrderedDict()
          layers["index"] = layer_index
          layers["values"] = [
              round(float(x), 6) for x in layer_output[i:(i + 1)].flat
          ]
          all_layers.append(layers)
        features = collections.OrderedDict()
        features["token"] = token
        features["layers"] = all_layers
        all_features.append(features)
      output_dict["features"] = all_features
      outputs.append(output_dict)
    
    return outputs[0]
  
  def extracts(self, sentences):
    
    examples = []
    for idx, sentence in enumerate(sentences):
      examples.append(InputExample(unique_id=idx, text_a=sentence, text_b=None))
    features = convert_examples_to_features(examples=examples,
                                                  seq_length=_max_seq_length,
                                                  tokenizer=self._tokenizer)
    unique_id_to_feature = {}
    for feature in features:
      unique_id_to_feature[feature.unique_id] = feature
    input_fn = input_fn_builder(features=features,
                                                  seq_length=_max_seq_length)
    outputs = []
    for idx, output in enumerate(self._estimator.predict(input_fn)):
      feature = unique_id_to_feature[idx]
      output_dict = collections.OrderedDict()
      all_features = []
      for (i, token) in enumerate(feature.tokens):
        all_layers = []
        for (j, layer_index) in enumerate(_layers):
          layer_output = output["layer_output_%d" % j]
          layers = collections.OrderedDict()
          layers["index"] = layer_index
          layers["values"] = [
              round(float(x), 6) for x in layer_output[i:(i + 1)].flat
          ]
          all_layers.append(layers)
        features = collections.OrderedDict()
        features["token"] = token
        features["layers"] = all_layers
        all_features.append(features)
      output_dict["features"] = all_features
      outputs.append(output_dict)
    
    return outputs

  def cal_dif_cls(self, emb1, emb2):
    dif = 0 
    for i in range(4):
      dif += (math.sqrt(sum((np.asarray(emb1['features'][0]['layers'][i]['values'])
                             - np.asarray(emb2['features'][0]['layers'][i]['values']))**2)/768))
    return dif
  
  def cal_dif_cls_layer(self, emb1, emb2, i):
    dif = (math.sqrt(sum((np.asarray(emb1['features'][0]['layers'][i]['values'])
                          - np.asarray(emb2['features'][0]['layers'][i]['values']))**2)/768))
    return dif
  
  def cal_dif_keyword(self, emb1, emb2, keyword):
    tokens = self._tokenizer.tokenize(keyword)
    candidates = []

    for emb in [emb1, emb2]:
      tmp_values = []
      sum_values = []
      for k, feature in enumerate(emb['features']):
        if feature['token'] == tokens[0]:
          tmp = []
          for t in range(len(tokens)):
            tmp.append(emb['features'][k+t]['layers'])
            if emb['features'][k+t]['token'] != tokens[t]:
              break
          if len(tmp) == len(tokens):
            tmp_values = [1] * 768
            sum_values = [tmp_values] * 4
            for _ in tmp:
              for i in range(4):
                for j in range(768):
                  sum_values[i][j] += _[i]['values'][j]
            for i in range(4):
              for j in range(768):
                sum_values[i][j] = sum_values[i][j]
            candidates.append(sum_values)
            break

    if len(candidates) < 2:
      return -1

    dif = 0
    for i in range(4):
      dif += (math.sqrt(sum((np.asarray(candidates[0][i])
                             - np.asarray(candidates[1][i]))**2)/768))
    return dif
  
if __name__ == "__main__":
  bert = BERT()  
  bert.init()

  sentences = ['‘세계의 공장’으로 막대한 달러를 쓸어담으며 경제력을 키웠던 중국의 좋은 시절도 오래가지 않을 듯>하다.',
            '자본 유출과 서비스 수지 적자 폭이 커지며 경상수지 적자를 향해 빠르게 다가가고 있어서다.',
             "[OBS 독특한 연예뉴스 조연수 기자] 가수 겸 배우 수지가 '국민' 타이틀을 거머쥔 스타로 꼽혔다.",
             "OBS '독특한 연예뉴스'(기획·연출·감수 윤경철, 작가 박은경·김현선)가 '국민 신드롬'을 일으킨 첫사랑의 아이콘 >김연아, 수지, 설현의 근황을 살펴봤다.",
             '오늘은 날씨가 좋습니다. 맛집을 찾아 가볼까요? 아이들이 좋아하더라구요.',
             '보쌈집에서는 보쌈을 맛있게 하면 그만입니다.ㅋㅋ']

  results = bert.extracts_(sentences)
  
  distances = []
  for i in range(len(results)):
    distance = []
    for j in range(len(results)):
      if i == j:
        distance.append(99999)
      else:
        distance.append(bert.cal_dif_cls(results[i], results[j]))
    distances.append(distance)

  for idx in range(len(sentences)):
    print(sentences[idx])
    print(sentences[distances[idx].index(min(distances[idx]))])
    print()
