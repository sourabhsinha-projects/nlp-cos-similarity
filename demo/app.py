import math
import pandas as pd
import numpy as np
from numpy.linalg import norm
from flask import Flask, request, render_template

# We will use TF-IDF algorithm as 
# TF-IDF minimizes the effect of stop-words and enhances the effect of meaningful words
# Term_Frequency -> tf(word) = (Number of times a word appear in a document)/(Total number of words in the document)
# Invesrse_Document_Frequency = idf(word) = log_e((number of documents)/(number of documents that contain the word))
# TF-IDF = tf(word)*idf(word)

def tf(wordHist, bow):
    tfHist = {}
    bowCnt = len(bow)
    for w, c in wordHist.items():
        tfHist[w] = c/bowCnt
    
    return tfHist

def idf(docLst):
    idfDct = {}
    totalDocs = len(docLst)
    
    # count the number of documents that contains a word
    idfDct = dict.fromkeys(docLst[0].keys(), 0)
    
    for d in docLst:
        for w, c in d.items():
            if c > 0:
                idfDct[w] += 1
    
    # Calculate idf score
    for w, c in idfDct.items():
        # Add a bias +1, as log(1) = 0 for same documents
        idfDct[w] = math.log(totalDocs/c) + 1
    
    return idfDct

def tfidf(tfBow, idfs):
    tfidf = {}
    for w, c in tfBow.items():
        tfidf[w] = c*idfs[w]
    
    return tfidf
    
app = Flask(__name__)

@app.route('/')
def text_similarity_form():
	return render_template('form.html')
	
@app.route('/', methods=['POST'])
def text_similarity_post():
	doc1 = request.form['doc1']
	doc2 = request.form['doc2']
	
	#Tokenize, remove punctuations and convert to same case
	bowDoc1 = [word.rstrip('.,;!? ').lower() for word in doc1.split()]
	bowDoc2 = [word.rstrip('.,;!? ').lower() for word in doc2.split()]
	
	# Do some more cleanup
	# Remove 's' from the end of the word, this is easiest. Removing 'ing' bit tricky, will leave rest for now
	bowDoc1 = [word.rstrip('s') for word in bowDoc1]
	bowDoc2 = [word.rstrip('s') for word in bowDoc2]
	
	# Expand short hand notation "'ll'" --> will, "n't'" and it's important to have these 
	# as e.g. the meaning of NOT is very different than no NOT
	
	tempList = []
	word1 = ''
	word2 = ''
	for word in bowDoc1:
	    if "\'ll" in word:
	        bowDoc1.remove(word)
	        word1 = word[:-3]
	        word2 = "will"
	        
	        tempList.append(word1)
	        tempList.append(word2)
	    
	    if "n\'t" in word:
	        bowDoc1.remove(word)
	        word1 = word[:-3]
	        word2 = "not"
	        
	        tempList.append(word1)
	        tempList.append(word2)
	
	bowDoc1.extend(tempList)

	tempList = []
	word1 = ''
	word2 = ''
	for word in bowDoc2:
	    if "\'ll" in word:
	        bowDoc2.remove(word)
	        word1 = word[:-3]
	        word2 = "will"
	        
	        tempList.append(word1)
	        tempList.append(word2)
	    
	    if "n\'t" in word:
	        bowDoc2.remove(word)
	        word1 = word[:-3]
	        word2 = "not"
	        
	        tempList.append(word1)
	        tempList.append(word2)
	
	bowDoc2.extend(tempList)
	
	# Create a combined bag-of-words
	bow = set(bowDoc1).union(set(bowDoc2))
	
	# Initiliaze the word-count Histograms
	wordHist1 = dict.fromkeys(bow, 0)
	wordHist2 = dict.fromkeys(bow, 0)

	# Fill up the count values
	for w in bowDoc1:
	    wordHist1[w] += 1
	
	for w in bowDoc2:
	    wordHist2[w] += 1
	    
	wordCntDf = pd.DataFrame([wordHist1, wordHist2])
	
	tfBow1 = tf(wordHist1, bowDoc1)
	tfBow2 = tf(wordHist2, bowDoc2)
	
	idfs = idf([wordHist1, wordHist2])
	
	tfidfBow1 = tfidf(tfBow1, idfs)
	tfidfBow2 = tfidf(tfBow2, idfs)
	
	tfidfDoc1 = wordCntDf.iloc[0].to_numpy()
	tfidfDoc2 = wordCntDf.iloc[1].to_numpy()
	
	cos_similarity = 100*round(np.dot(tfidfDoc1, tfidfDoc2)/(norm(tfidfDoc1)*norm(tfidfDoc2)), 4)
	
	return render_template('form.html', result=cos_similarity, doc1=doc1, doc2=doc2)

if __name__ == "__main__":
	app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)