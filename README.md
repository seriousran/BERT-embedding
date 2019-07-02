# BERT-embedding
A simple wrapper class for extracting features(embedding) and comparing them using BERT

## How to Use

### Installation
```bash
git clone https://github.com/seriousmac/BERT-embedding.git
cd BERT-embedding
pip install -r requirements.txt
wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
unzip multi_cased_L-12_H-768_A-12.zip -d bert/
```

### Run Test
```bash
python bert_embedding.py
```

### Example - 한 문장에서 embedding 추출하기
```python
from bert_embedding import BERT

bert = BERT()
bert.init()

sentence = "[OBS 독특한 연예뉴스 조연수 기자] 가수 겸 배우 수지가 '국민' 타이틀을 거머쥔 스타로 꼽혔다."
result = bert.extracet(sentence)
```

### Example 2 - 여러 문장에서 embedding 추출하기
```python
from bert_embedding import BERT

bert = BERT()
bert.init()

sentences = ['‘세계의 공장’으로 막대한 달러를 쓸어담으며 경제력을 키웠던 중국의 좋은 시절도 오래가지 않을 듯>하다.',
   '자본 유출과 서비스 수지 적자 폭이 커지며 경상수지 적자를 향해 빠르게 다가가고 있어서다.',
   "[OBS 독특한 연예뉴스 조연수 기자] 가수 겸 배우 수지가 '국민' 타이틀을 거머쥔 스타로 꼽혔다.",
   "OBS '독특한 연예뉴스'(기획·연출·감수 윤경철, 작가 박은경·김현선)가 '국민 신드롬'을 일으킨 첫사랑의 아이콘 >김연아, 수지, 설현의 근황을 살펴봤다."]
results = bert.extracts(sentences)
```
