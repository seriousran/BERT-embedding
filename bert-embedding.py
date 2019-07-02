import os
import sys
import tensorflow as tf
sys.path.append("bert")
import modeling, tokenization
from extract_features import input_fn_builder, model_fn_builder, convert_examples_to_features, InputExample

# https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
DIR = 'bert/'
MODEL_DIR  = DIR + 'multi_cased_L-12_H-768_A-12/'
CONFIG = MODEL_DIR + 'bert_config.json'
CKPT  = MODEL_DIR + 'bert_model.ckpt'
VOCAB  = MODEL_DIR + 'vocab.txt'

_max_seq_length = 512
_layers = [-1, -2, -3, -4]

class BERT():
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

  def extract(self, sentence):
    
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
    
  
  def extracts(self, sentences):
    
    examples = []
    for sentence in sentences:
      examples.append(InputExample(unique_id=0, text_a=sentence, text_b=None))
    features = convert_examples_to_features(examples=examples,
                                                  seq_length=_max_seq_length,
                                                  tokenizer=self._tokenizer)
    input_fn = input_fn_builder(features=features,
                                                  seq_length=_max_seq_length)
    outputs = []
    for output in self._estimator.predict(input_fn):
      outputs.append(output)
    
    return outputs

if __name__ == "__main__":
  bert = BERT()  
  bert.init()

  sentences = ['‘세계의 공장’으로 막대한 달러를 쓸어담으며 경제력을 키웠던 중국의 좋은 시절도 오래가지 않을 듯하다.',
   '자본 유출과 서비스 수지 적자 폭이 커지며 경상수지 적자를 향해 빠르게 다가가고 있어서다.',
   "[OBS 독특한 연예뉴스 조연수 기자] 가수 겸 배우 수지가 '국민' 타이틀을 거머쥔 스타로 꼽혔다.",
   "OBS '독특한 연예뉴스'(기획·연출·감수 윤경철, 작가 박은경·김현선)가 '국민 신드롬'을 일으킨 첫사랑의 아이콘 김연아, 수지, 설현의 근황을 살펴봤다."]

  #for sentences in sentences:
  print(bert.extracts(sentences))