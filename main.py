import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

# 사전 학습된 모델과 토크나이저 로드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS).to(device)

# 웹사이트 URL
url = 'https://www.example.com'

# 요청(Request) 보내기
response = requests.get(url)

# BeautifulSoup 객체 생성 및 HTML 파싱
soup = BeautifulSoup(response.text, 'html.parser')

# 원하는 정보 선택하기 (여기서는 페이지 전체의 텍스트를 추출)
text = soup.get_text()

# 입력 데이터 준비
inputs = tokenizer(text[:512], truncation=True, padding=True, return_tensors='pt').to(device)

# 예측 수행
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class_idx = torch.argmax(outputs.logits).item()

print(f"The predicted class index is: {predicted_class_idx}")
