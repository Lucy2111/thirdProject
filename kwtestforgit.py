from newspaper import Article
from konlpy.tag import Kkma
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np

class SentenceTokenizer(object):
  def __init__(self):
    self.kkma = Kkma()
    self.okt = Okt()
    self.stopwords = ["중인" ,"만큼", "작품", "같은", "다른", "부터", "이것", "저것", "그것", "작가", "독자", "소재", "통해", "위해", "이해", "무엇", "어떤", "사실", "때문" "마찬가지", "책소개", "저자", "저자소개", "추천의 글", "목차", "출판사", "서평" ,"아", "과", "와", "때문이다", "리뷰", "어", "나", "우리", "따라", "의해", "위해" "을", "를", "에", "의", "가", "한다", "되다", "된다", "이다", "했다", "였다", "었다"]
  def url2sentences(self, url):
    article = Article(url, language='ko')
    article.download()
    article.parse()
    sentences = self.kkma.sentences(article.text)
    for idx in range(0, len(sentences)):
      if len(sentences[idx]) <= 10:
        sentences[idx-1] += (' ' + sentences[idx])
        sentences[idx] = ''
    return sentences
  def text2sentences(self, text):
      sentences = self.kkma.sentences(text)
      for idx in range(0, len(sentences)):
        if len(sentences[idx]) <= 10:
          sentences[idx-1] += (' ' + sentences[idx])
          sentences[idx] = ''
      return sentences
  def get_nouns(self, sentences):
    nouns = []
    for sentence in sentences:
      if sentence is not '':
        nouns.append(' '.join([noun for noun in self.okt.nouns(str(sentence))
          if noun not in self.stopwords and len(noun) > 1]))
    return nouns

class GraphMatrix(object):
  def __init__(self):
    self.tfidf = TfidfVectorizer()
    self.cnt_vec = CountVectorizer()
    self.graph_sentence = []
  def build_sent_graph(self, sentence):
    tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
    self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
    return self.graph_sentence
  def build_words_graph(self, sentence):
    cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
    vocab = self.cnt_vec.vocabulary_
    return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}

class Rank(object):
  def get_ranks(self, graph, d=0.85): # d = damping factor
    A = graph
    matrix_size = A.shape[0]
    for id in range(matrix_size):
        A[id, id] = 0 # diagonal 부분을 0으로
        link_sum = np.sum(A[:,id]) # A[:, id] = A[:][id]
        if link_sum != 0:
          A[:, id] /= link_sum
        A[:, id] *= -d
        A[id, id] = 1
    B = (1-d) * np.ones((matrix_size, 1))
    ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
    return {idx: r[0] for idx, r in enumerate(ranks)}

class TextRank(object):
  def __init__(self, text):
    self.sent_tokenize = SentenceTokenizer()
    if text[:5] in ('http:', 'https'):
      self.sentences = self.sent_tokenize.url2sentences(text)
    else:
      self.sentences = self.sent_tokenize.text2sentences(text)
    self.nouns = self.sent_tokenize.get_nouns(self.sentences)
    self.graph_matrix = GraphMatrix()
    self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
    self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)
    self.rank = Rank()
    self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
    self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
    self.word_rank_idx = self.rank.get_ranks(self.words_graph)
    self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
  def summarize(self, sent_num=3):
    summary = []
    index=[]
    for idx in self.sorted_sent_rank_idx[:sent_num]:
      index.append(idx)
    index.sort()
    for idx in index:
      summary.append(self.sentences[idx])
    return summary
  def keywords(self, word_num=5):
    rank = Rank()
    rank_idx = rank.get_ranks(self.words_graph)
    sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)
    keywords = []
    index=[]
    for idx in sorted_rank_idx[:word_num]:
      index.append(idx)
    #index.sort()
    for idx in index:
        keywords.append(self.idx2word[idx])
    return keywords

import os
import sys
import urllib.request
client_id = "" # 개발자센터에서 발급받은 Client ID 값
client_secret = "" # 개발자센터에서 발급받은 Client Secret 값

#api
from flask import Flask, request as freq, jsonify  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
app.config['JSON_AS_ASCII'] = False

@app.route('/getkeyword')  # 데코레이터 이용, '/getkeyword' 경로에 클래스 등록
def test():
    url = freq.args.get('link', None)
    
    if url is not None:
      encText = urllib.parse.quote(url)
      data = "url=" + encText
      req = urllib.request.Request(url)
      req.add_header("X-Naver-Client-Id",client_id)
      req.add_header("X-Naver-Client-Secret",client_secret)
      response = urllib.request.urlopen(req, data=data.encode("utf-8"))
      rescode = response.getcode()
      textrank = TextRank(url)

      if(rescode==200):
          data = {"success":True, "data":textrank.keywords()}
          return jsonify(data), 200
      else:
          data = {"success":False, "data": []}
          return jsonify(data), 500
    else:
        data = {"success":False, "data": []}
        return jsonify(data), 500

    
   
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)