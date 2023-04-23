


**Problem Statement**

CISI, or the "Cranfield collection," is a standard dataset used in Information Retrieval research. It consists of documents, queries, and relevance assessments that can be used to evaluate and compare different retrieval systems. The problem statement for CISI can be framed as follows: given a set of queries and a collection of documents, the goal is to develop an information retrieval system that can effectively rank the documents in order of their relevance to the query. The relevance of each document to each query is already provided in the dataset, allowing for evaluation of the system's effectiveness. The CISI dataset contains 1460 documents, 112 queries, and 9 relevance assessments per query, for a total of 100 queries with relevance judgments. The documents cover a wide range of topics, including medicine, engineering, and social sciences, and the queries reflect information needs that a user might have in these fields.

**Motivation**

The CISI dataset is a collection of documents and queries used for information retrieval research. The motivation behind creating this dataset was to provide researchers with a standard benchmark for evaluating information retrieval systems. Prior to the creation of CISI, there were no widely accepted benchmarks for information retrieval systems. This made it difficult for researchers to compare the effectiveness of different retrieval algorithms and techniques. The CISI dataset was designed to fill this gap by providing a common set of 

documents and queries that researchers could use to evaluate their systems. The CISI dataset 



consists of 1460 abstracts from articles in the field of information science, along with a set of 112 queries that were used to retrieve relevant documents from the collection. The queries cover a variety of topics in information science, including information retrieval, indexing, classification, and more. By providing a standard benchmark for information retrieval systems, the CISI dataset has helped to drive advances in the field of information retrieval. Researchers can use this dataset to test new algorithms and techniques, and to compare their results to those of other researchers. This has led to a better understanding of the strengths and weaknesses of different approaches to information retrieval, and has helped to improve the performance of information retrieval systems in general.

**Literature review**

1. "Neural Information Retrieval: A Literature Review" by Kezban Dilek Onal, Ye Zhang, Ismail Sengor Altingovde, and Pinar Karagoz. This paper provides a comprehensive review of recent developments in neural information retrieval, including deep learning models, attention mechanisms, and transfer learning. The paper also discusses the challenges and opportunities in applying these techniques to real-world retrieval systems. 
1. "A Survey of Learning to Rank for Information Retrieval" by Tie-Yan Liu. This paper provides an overview of learning to rank techniques for information retrieval, including pointwise, pairwise, and listwise approaches. The paper also discusses the advantages and disadvantages of these approaches, and provides insights into the current state of the art. 
1. "BM25F Revisited" by Nikolaos Pappas, Giannis Nikolentzos, and Michalis Vazirgiannis. This paper proposes a new variant of the popular BM25 algorithm for document ranking, which takes into account the relevance of different fields in a document. The paper shows that the new algorithm outperforms the original BM25 algorithm on several datasets, including the CISI dataset. 
1. "Query Expansion Techniques for Information Retrieval: A Comprehensive Survey" by Aditya Gupta and Anurag Jain. This paper provides a comprehensive survey of query expansion techniques for information retrieval, including pseudo-relevance feedback, relevance models, and thesaurus-based methods. The paper also discusses the advantages and disadvantages of these techniques, and provides insights into the current state of the art.
1. "An Evaluation of Query Performance Prediction Methods" by Joao Magalhaes and Maarten de Rijke. This paper compares different methods for predicting the performance of information retrieval queries, including machine learning and statistical methods. The paper shows that some of the proposed methods outperform traditional relevance models on several datasets, including the CISI dataset

Overall, there is a lot of research and development in the field of information retrieval, and many promising techniques have been proposed. The choice of approach will depend on the specific requirements of the problem, such as the size of the dataset, the nature of the queries, and the desired level of accuracy.

**Novelty**

1. Using TF-ICF instead of TF-IDF Ranking Model.
1. Using Word2Vec, Spacy, GloVe as embeddings.
1. Using USE instead of Sentence BERT.
1. Using a Pre-Trained BERT Model and Fine-Tuning it to our system. **all-MiniLM-L6-v2.** 

**Methodology**

**Procedure**

1. Collect and preprocess the documents: Collect the relevant documents and preprocess them by cleaning, tokenizing, and normalizing the text. This can involve removing stop words, stemming or lemmatization, and other text preprocessing techniques.
1. Index the documents: Create an index that maps each document to its unique identifier, and stores the relevant information about the document, such as its title, content, author, and other metadata.
1. Query processing: When the user enters a query, preprocess the query by tokenizing, cleaning, and normalizing the text. Then, use the index to retrieve the documents that are most relevant to the query. This can involve using techniques such as Boolean search, vector space model, or other information retrieval models.
1. Document ranking and presentation: Once the relevant documents are retrieved, rank them according to their relevance to the query. This can be done using techniques such as TF-IDF, BM25, or other ranking algorithms. Then, present the results to the user in a way that is easy to understand and navigate, such as a list of documents with their titles, authors, and other relevant information.

**TF-IDF Ranking Model**

TF-IDF (Term Frequency-Inverse Document Frequency) is a technique used to weigh the importance of words in a document or corpus of documents. It takes into account both the frequency of a term in a document and the frequency of the term across all documents in the corpus. 

Here are the steps to build a TF-IDF Ranking Model :-

1. Collect and preprocess the documents: Collect the relevant documents and preprocess them by cleaning, tokenizing, and normalizing the text. This can involve removing stop words, stemming or lemmatization, and other text preprocessing techniques.
1. Compute the TF-IDF scores: Compute the term frequency-inverse document frequency (TF-IDF) scores for each term in each document. TF-IDF is a measure that reflects the importance of a term in a document relative to its importance in the entire corpus. The TF-IDF score for a term t in a document d is computed as follows: TF-IDF(t, d) = TF(t, d) \* IDF(t)
1. where TF(t, d) is the frequency of term t in document d, and IDF(t) is the inverse document frequency of term t, computed as: IDF(t) = log(N/df(t))
1. where N is the total number of documents in the corpus, and df(t) is the number of documents that contain the term t.
1. Create an index: Create an index that maps each term to the documents that contain it and its corresponding TF-IDF score. This can be done using a dictionary or inverted index data structure.
1. Query processing: When the user enters a query, preprocess the query by tokenizing, cleaning, and normalizing the text. Then, use the index to retrieve the documents that are most relevant to the query. This can involve using techniques such as Boolean search, vector space model, or other information retrieval models.
1. Document ranking and presentation: Once the relevant documents are retrieved, rank them according to their relevance to the query using the TF-IDF scores. This can be done by computing a score for each document based on the TF-IDF scores of the query terms that appear in the document. Then, present the results to the user in a way that is easy to understand and navigate, such as a list of documents with their titles, authors, and other relevant information.

Overall, creating a document information retrieval system using TF-IDF ranking involves collecting and preprocessing the documents, computing the TF-IDF scores, creating an index, processing queries, and ranking and presenting the results based on the TF-IDF scores. The specific techniques and algorithms used will depend on the requirements and constraints of the particular system.

**TF-ICF Ranking Model**

TF-ICF (Term Frequency-Inverse Category Frequency), which is a variant of the TF-IDF technique that takes into account the frequency of a term within a specific category instead of the entire corpus. Here are the steps to perform TF-ICF: -

1. Preprocessing: First, you need to preprocess the text data by removing stop words, punctuation, and performing stemming or lemmatization if required.
1. Term Frequency (TF): Calculate the frequency of each word in a document. Term Frequency can be calculated by dividing the number of times a word appears in a document by the total number of words in the document. TF = (Number of times the term appears in a document) / (Total number of terms in the document)
1. Term Frequency (TF): Calculate the frequency of each word in a document. Term Frequency can be calculated by dividing the number of times a word appears in a document by the total number of words in the document. TF = (Number of times the term appears in a document) / (Total number of terms in the document)
1. TF-ICF: Finally, you can calculate the TF-ICF score for each word in each document by multiplying the term frequency and the inverse category frequency. TF-ICF = TF \* ICF
1. Then, create an index that maps each term to the documents that contain it and its corresponding TF-ICF score. This can be done using a dictionary or inverted index data structure. Finally, use this index to retrieve the documents that are most relevant to a given query, and rank them based on their TF-ICF scores.

The resulting TF-ICF score will be higher for words that are more frequent in the document but rare within a specific category, indicating that these words are more important for that document within that category. Conversely, the score will be lower for words that are less frequent in the document or more common within that category, indicating that they are less important for that document within that category. 

**Spacy or Glove Embeddings**

1. Preprocess your data: Clean, tokenize, and normalize your data, and prepare it for use with Spacy or GloVe. 
1. Vectorize your documents: Convert each document into a vector representation using Spacy or GloVe. Spacy and GloVe both provide pre-trained word embeddings that can be used to convert words into vectors. To vectorize a document, you can either take the average of the word vectors or use more advanced techniques such as TF-IDF weighting or document embeddings. 
1. Build an index: Create an index that maps the vector representation of each document to its ID. This index can be stored in memory or on disk, depending on the size of your dataset. 
1. Query processing: When a user enters a query, vectorize the query using the same technique used for vectorizing the documents. Then, use the index to retrieve the top-k documents that are most similar to the query. The similarity between the query and the documents can be measured using cosine similarity or other distance metrics. 
1. Ranking: Once you have retrieved the top-k documents, rank them according to their relevance to the query. This can be done using a variety of techniques, such as BM25, language models, or machine learning models.

**Word2Vec Ranking Model**

1. Data preprocessing: The first step is to prepare the data for training the word2vec model. This includes cleaning the text data by removing stop words, punctuations, and other noise from the text corpus. The data also needs to be tokenized, which means breaking down the text into individual words.
1. Training the word2vec model: Once the data is preprocessed, the next step is to train the word2vec model. This involves passing the tokenized text data through the word2vec model to create vector representations of each word in the corpus. These vector representations are learned by the model and can capture the semantic and contextual meaning of each word in the text data.
1. Indexing the documents: After the word2vec model is trained, the next step is to index the documents in the system. This can be done using a search engine or other indexing tools.
1. Retrieval process: Once the documents are indexed, the retrieval process can begin. When a user enters a query, the query text is tokenized and each token is passed through the word2vec model to generate vector representations of the words in the query. These vector representations are then compared with the vector representations of each document in the corpus to find the most relevant documents. The cosine similarity is often used to calculate the similarity between two vectors, with a higher cosine similarity indicating a closer match.
1. Ranking the results: Once the relevant documents are retrieved, they need to be ranked based on their relevance to the query. This can be done using a variety of ranking algorithms, such as the BM25 algorithm or the TF-IDF algorithm.



**Using USE(universal sentence encoder)**

The USE model uses a deep neural network that consists of multiple layers of self-attention and feed-forward networks. The self-attention mechanism allows the model to focus on different parts of the input sentence, while the feed-forward networks transform the input into a high-dimensional vector representation. The main reason for trying USE was the variety of data used in its training. However the exact corpus size of USE was not available Which can give hard time for comparing it with other models. There are not many technical details regarding the USE model. 

The code for the following is given in the zip. 

1. There can be some possibilities for the above lower results. 
1. Since parameters of USE are greater than BERT, this can happen that the corpus of USE is smaller than bert.
1. BERT architecture can more good than USE
1. BERT training can be more oriented toward our dataset as USE is trained on large variety of dataset There is more research on BERT and many more additional models are built on BERT and due to this updated of bert can lead to better result

**Fine Tuned BERT Model**

The following steps are typically followed to fine-tune a BERT model: 

1. Prepare the data: The first step is to prepare the task-specific data for training. This involves collecting and preprocessing the data, splitting it into training and validation sets, and converting the text data into numerical form. 
1. Load the pre-trained BERT model: The next step is to load a pre-trained BERT model that has been trained on a large corpus of text data. The pre-trained BERT model is typically loaded using a library like TensorFlow or PyTorch. 
1. Modify the model for the specific task: The pre-trained BERT model is then modified by adding a task-specific layer on top of the model. The task-specific layer can be a simple feedforward neural network or a more complex architecture, depending on the task being performed. 
1. Fine-tune the model: The modified BERT model is then trained on the task-specific data. During training, the weights of the pre-trained BERT model are updated using backpropagation. The model is typically trained for several epochs until the loss on the validation set stops decreasing. 
1. Evaluate the model: After training, the performance of the fine-tuned BERT model is evaluated on a held-out test set. The evaluation metrics can vary depending on the task, but commonly used metrics include accuracy, precision, recall, and F1 score. 


**Applying Fine Tuned Bert Model on Preprocessed & Clean Text**

Preprocessing and Text Cleaning 

1. Convert all text to lowercase
1. Remove punctuation
1. Remove numbers
1. Remove stop words 
1. Stemming/Lemmatization
1. Spell correction: 
1. Remove special characters: Special characters such as "$" or "@"

**Database**

[**CISI Dataset**](https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval)

Number of documents = 1460 

18 Editions of the Dewey Decimal Classifications Comaromi, J.P. The present study is a history of the DEWEY Decimal Classification. The first edition of the DDC was published in 1876, the eighteenth edition in 1971, and future editions will continue to appear as needed. In spite of the DDC's long and healthy life, however, its full story has never been told. There have been biographies of Dewey that briefly describe his system, but this is the first attempt to provide a detailed history of the work that more than any other has spurred the growth of librarianship in this country and abroad. 

Number of queries = 112 

Number of mappings = 76 ['28', '35', '38', '42', '43', '52', '65', '76', '86', '150', '189', '192', '193', '195', '215', '269', '291', '320', '429', '465', '466', '482', '483', '510', '524', '541', '576', '582', '589', '603', '650', '680', '711', '722', '726', '783', '813', '820', '868', '869', '894', '1162', '1164', '1195', '1196', '1281']

**Code**
**


tokened\_reviews = [] data\_review=data.Review.copy() for i in range(len(data\_review)): print(data\_review[i].split()) tokened\_reviews.append(data\_review[i].split()) model= gensim.models.Word2Vec(tokened\_reviews, min\_count = 1, size = 300, window = 5, sg = 1) sent\_vect\_all=[] for j in range(len(tokened\_reviews)): sum\_vect=model.wv[tokened\_reviews[j][0]] for i in tokened\_reviews[j]: sum\_vect=sum\_vect+model.wv[i] sent\_vect\_all.append(sum\_vect/len(tokened\_reviews[j]))


**Evaluation**

**Baseline Results**

1. For TF-IDF : 
   1. MRR@10 = 0.4240
1. For USE :
   1. Recall@10 = 0.096
   1. Precision@10 = 0.274
   1. MRR(Mean Reciprocal Rank) = 0.555
   1. MAP(Mean Average Precision)  =  0.051
1. For Fine Tuned BERT: 
   1. Recall@10 = 0.009
   1. Precision@10 = 0.595 
   1. Mean Reciprocal Rank (MRR) = 0.818
   1. MAP(Mean Average Precision)  =  0.192

**Our Results**

Note: Initial Models were Discarded.

After Preprocessing, Text Cleaning and Fine Tuning the model with more epochs and  focused more on cleaned data.

**For Fine Tuned BERT**

1. Recall@10 = 0.009
1. Precision@10 = 0.721
1. Mean Reciprocal Rank (MRR) = 0.883
1. MAP(Mean Average Precision)  =  0.282


**Contributions**

Equal contributions by all members - writing problem statement , cleaning & preprocessing text , understanding the process of solving problem , applying USE model , fine tuning the initial BERT model, Re-Training the model with cleaned text, ,report making, Powerpoint presentation

**References**

1. <https://arxiv.org/ftp/arxiv/papers/1012/1012.2609.pdf>
1. https://arxiv.org/pdf/1803.11175.pdf
1. <https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a>
1. <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>
1. <https://www.kaggle.com/code/rid17pawar/universal-sentence-encoder>
1. <https://www.kaggle.com/code/rid17pawar/sentence-bert>
1. tensorflow.org/hub/tutorials/semantic\_similarity\_with\_tf\_hub\_universal\_encoder


