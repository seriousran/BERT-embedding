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
python bert-embedding.py
```

### Example
```python
from bert_embedding import BERT

bert = BERT()
bert.init()

sentence = "[OBS 독특한 연예뉴스 조연수 기자] 가수 겸 배우 수지가 '국민' 타이틀을 거머쥔 스타로 꼽혔다."
result = bert.extracet(sentence)
```
