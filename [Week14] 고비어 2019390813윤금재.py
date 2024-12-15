#!/usr/bin/env python
# coding: utf-8

# In[9]:


# 0번 ------------------------------------------------------------------
get_ipython().system('pip install langchain')
get_ipython().system('pip install langchain-community')
get_ipython().system('pip install langchain-huggingface')
get_ipython().system('pip install tiktoken')
get_ipython().system('pip install chromadb')
get_ipython().system('pip install sentence-transformers')


# In[2]:


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from typing import List
from langchain.schema import Document  # Import Document if needed
import pandas as pd


# In[12]:


# Document Load
loader = TextLoader('/Users/yoonchanghoon/Desktop/Intro.txt')
data = loader.load()


# In[13]:


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=150,
    chunk_overlap=50,
    encoding_name='cl100k_base'
)

texts = text_splitter.split_text(data[0].page_content)


# In[14]:


texts


# In[10]:


pip install sentence-transformers


# In[15]:


embeddings_model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sroberta-nli',
        # huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':True}
    )


# In[16]:


print(texts[0])
embeddings_model.embed_query(texts[0])[:3]


# In[17]:


# Vectorized size
len(embeddings_model.embed_query(texts[0]))


# In[18]:


db = Chroma.from_texts(
    texts, 
    embeddings_model,
    collection_name = 'history',
    persist_directory = './db/chromadb',
    collection_metadata = {'hnsw:space': 'cosine'}, # l2 is the default
)

db


# In[19]:


query = '디지털경영전공은 무엇인가요?'
docs = db.similarity_search(query)
print(docs[0].page_content)


# In[20]:


docs


# In[21]:


query = '데이터분석과 빅데이터는 무슨 상관인가요'
docs = db.similarity_search(query)
print(docs[0].page_content)


# In[ ]:





# In[22]:


#01번-------------------------------------------------------------------


# In[36]:


import pandas as pd 
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertModel


# In[37]:


# Hugging Face 모델 로드 (GPT-2와 BERT)
# GPT-2 모델 (텍스트 생성용)
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# BERT 모델 (임베딩용)
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# In[38]:


# Read the online retail dataset
data = pd.read_csv('/Users/yoonchanghoon/Desktop/amazon(1).csv')
df = data[:100].copy()
df.dropna(subset=['rating_count'], inplace=True)

df['sub_category'] = df['category'].astype(str).str.split('|').str[-1]
df['main_category'] = df['category'].astype(str).str.split('|').str[0]


# In[39]:


df.columns


# In[40]:


df1 = df.copy()
df1['product_name'] = df1['product_name'].str.lower() 
df1 = df1.drop_duplicates(subset=['product_name']) 


# In[41]:


print(df.shape)
print(df1.shape)


# In[42]:


df1['product_name'][0]


# In[43]:


df1['about_product']


# In[44]:


df2 = df1[['product_id','product_name', 'about_product','main_category','sub_category', 'actual_price','discount_percentage','rating','rating_count' ]]


# In[45]:


df2.head()


# In[47]:


import pandas as pd
import os

# 예시: 제품 관련 질문-답변 데이터
data = {
    "question": [
        "What is the product's rating?",
        "How long is the product warranty?",
        "Is this product available in different colors?"
    ],
    "answer": [
        "4.5 stars based on 200 reviews.",
        "The warranty is 2 years.",
        "Yes, it is available in blue, red, and green."
    ],
    "context": [
        "Product details: Rating: 4.5 stars. Review count: 200.",
        "Product details: Warranty: 2 years.",
        "Product details: Available colors: blue, red, green."
    ]
}

# DataFrame 생성
df2 = pd.DataFrame(data)

# 'data' 폴더가 없으면 생성
if not os.path.exists('data'):
    os.makedirs('data')

# df2를 'amazon_rag.csv'로 저장
df2.to_csv('data/amazon_rag.csv', index=False)

# 결과 확인
print(df2)

from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# CSV 파일 로드
csv_loader = CSVLoader(file_path="data/amazon_rag.csv")
csv_docs = csv_loader.load()

# 디렉토리 내 모든 .md 파일 로드
directory_loader = DirectoryLoader("../", glob="**/*.md", loader_cls=TextLoader)
directory_docs = directory_loader.load()

# 로드된 문서 출력
print(csv_docs)
print(directory_docs)



# In[48]:


from typing import List
from dotenv import load_dotenv
import os

from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Chroma
from langchain.schema.document import Document


# This will expose your Langchain api token as an environment variable
load_dotenv()

# def read_csv(file_path: str, source_column: str = "about_product") -> List[Document]:
def read_csv(file_path: str, source_column: str = "product_name") -> List[Document]:
    """Reads a CSV file and returns a list of Documents.

    Args:
        file_path (str): The path to the CSV file to read.
        source_column (str, optional): The name of the column in the CSV file that contains the text data. Defaults to "Description".

    Returns:
        List[Document]: A list of Documents, where each Document contains the text data from the corresponding row in the CSV file.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        IOError: If there is an error reading the CSV file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    loader = CSVLoader(file_path=file_path, source_column=source_column)
    data = loader.load()

    return data


# In[52]:


import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Hugging Face API 토큰 환경 변수 설정
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # 본인의 토큰으로 변경

# 모델 이름 지정
model_name = 'sentence-transformers/all-mpnet-base-v2'

# Hugging Face 모델 로드
def load_embeddings_model(model_name: str) -> HuggingFaceEmbeddings:
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_function

# 임베딩 모델 로드
embeddings = load_embeddings_model(model_name)

# 텍스트 예시
text = "This is a test document."
embedding = embeddings.embed_query(text)
print(embedding)


# In[50]:





# In[53]:


model_name = 'sentence-transformers/all-mpnet-base-v2'
# model_name = 'jhgan/ko-sroberta-nli'
embeddings = load_embeddings_model(model_name)

# Test embedding a query
text = "This is a test document."
query_result = embeddings.embed_query(text)
print(query_result[:3])


# In[54]:


def vectorize_documents(data: List[Document], embedding_function: HuggingFaceEmbeddings) -> Chroma:
    """Vectorizes a list of Documents using a Hugging Face Transformer model.

    Args:
        data (List[Document]): A list of Documents to vectorize.
        embedding_function (HuggingFaceEmbeddings): An Embeddings object that can be used to encode text into embeddings.

    Returns:
        Chroma: A Chroma object that contains the vectorized documents.
    """

    ## Chroma, as a vector database, cosine similarity by default for searches.
    db = Chroma.from_documents(data, embedding=embedding_function, 
                            #    collection_metadata={"hnsw:space": "l2"}
                               collection_metadata={"hnsw:space": "cosine"}
                               )
    return db


# In[61]:


def init_llm():
    """Initializes the LLM by reading the CSV file, loading the embeddings model, and vectorizing the documents.

    Returns:
        Chroma: A Chroma object that contains the vectorized documents.
    """
    # Replace 'read_csv' with the appropriate CSV reading logic
    data = read_csv(file_path='data/amazon_rag.csv', source_column="question")
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    # model_name = 'jhgan/ko-sroberta-nli'
    embedding_function = load_embeddings_model(model_name)
    db = vectorize_documents(data, embedding_function)
    return db


# In[62]:


db = init_llm()


# In[64]:


# Query the vector database
query = "iPhone USB charger and adapter"
found_docs = db.similarity_search_with_score(query, k=5)


# In[65]:


# Load documents
found_docs


# In[66]:


# Document with Score
document, score = found_docs[-1]
print(document.page_content)
print(f"\nScore: {score}")


# In[67]:


found_docs[-1][0].page_content


# In[68]:


#Get source 
found_docs[0][0].metadata['source']


# In[69]:


# def run_vector_search(query: str):
#     # print(query)
#     query_vector = db.similarity_search_with_score(query, k=5)
#     # print(query_vector)
#     # document, score = query_vector
    
#     # return query_vector.metadata['source'], document.page_content, score
#     return query_vector

def run_vector_search(query: str, k = 3) -> str:
    """Performs a vector search and returns results in a structured format."""
    query_vector = db.similarity_search_with_score(query, k=k)

    # Extract and structure the search results
    results = []
    for document, score in query_vector:
        # Extract metadata and page content
        metadata = document.metadata
        page_content = {k.strip(): v.strip() for k, v in [line.split(':', 1) for line in document.page_content.split('\n') if ':' in line]}
        
        # Combine all data into a single dictionary
        combined_result = {
            "source": metadata.get("source", "Unknown Source"),
            **page_content,
            "score": score
        }
        results.append(combined_result)

    # Convert to a DataFrame
    df_results = pd.DataFrame(results)
    return df_results


# In[70]:


run_vector_search('what is the iphone cable?')


# In[76]:


pip install --upgrade accelerate


# In[82]:


pip install 'accelerate>=0.26.0'


# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


get_ipython().system('pip uninstall accelerate')



# In[90]:


get_ipython().system('pip install accelerate')


# In[93]:


import os
from transformers import AutoTokenizer
import transformers
import torch
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from dotenv import load_dotenv
from typing import List

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # 본인의 토큰으로 변경

# 모델 이름
model_name = 'sentence-transformers/all-mpnet-base-v2'

# Hugging Face 모델 로드
def load_embeddings_model(model_name: str) -> HuggingFaceEmbeddings:
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_function

# 임베딩 모델 로드
embeddings = load_embeddings_model(model_name)

# 텍스트 예시
text = "This is a test document."
embedding = embeddings.embed_query(text)
print(embedding)

# CSV 파일 로드
def read_csv(file_path: str, source_column: str = "product_name") -> List[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    loader = CSVLoader(file_path=file_path, source_column=source_column)
    data = loader.load()

    return data

# Chroma로 문서 벡터화
def vectorize_documents(data: List[Document], embedding_function: HuggingFaceEmbeddings) -> Chroma:
    db = Chroma.from_documents(data, embedding=embedding_function, collection_metadata={"hnsw:space": "cosine"})
    return db

# LLM 초기화
def init_llm():
    data = read_csv(file_path='data/amazon_rag.csv', source_column="question")
    embedding_function = load_embeddings_model(model_name)
    db = vectorize_documents(data, embedding_function)
    return db

# Query를 통해 벡터 검색
def run_vector_search(query: str, k = 3) -> str:
    query_vector = db.similarity_search_with_score(query, k=k)
    results = []
    for document, score in query_vector:
        metadata = document.metadata
        page_content = {k.strip(): v.strip() for k, v in [line.split(':', 1) for line in document.page_content.split('\n') if ':' in line]}
        combined_result = {
            "source": metadata.get("source", "Unknown Source"),
            **page_content,
            "score": score
        }
        results.append(combined_result)
    
    df_results = pd.DataFrame(results)
    return df_results

# 모델 초기화 및 실행
db = init_llm()

# 예시로 "what is the iphone cable?" 검색
found_docs = run_vector_search('what is the iphone cable?')
print(found_docs)


# In[ ]:





# In[96]:


pip install 'accelerate>=0.26.0'


# In[ ]:





# In[98]:


pip install --upgrade transformers


# In[99]:


pipeline = transformers.pipeline(
    "text-generation",  
    model=model,    
    tokenizer=tokenizer,  
    torch_dtype=torch.float32,  # float16을 사용하지 않으면 에러 방지
    device=-1,  # CPU에서 실행 (GPU가 없다면)
)


# In[101]:


get_ipython().system('accelerate config')


# In[105]:


from transformers import AutoTokenizer
import transformers 
import torch

# Load model
# model = "nilq/mistral-1L-tiny" 
#huggingface-cli login
#Reference: https://huggingface.co/docs/huggingface_hub/en/guides/cli

model = "ArliAI/Gemma-2-2B-ArliAI-RPMax-v1.1"  # 사용할 모델 이름
tokenizer = AutoTokenizer.from_pretrained(
    model,  
    use_auth_token='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'  # 본인의 토큰을 이곳에 입력
)

# pipeline 
pipeline = transformers.pipeline(
    "text-generation",  
    model=model,    
    tokenizer=tokenizer,  # 모델 로드 후 토크나이저 설정
    torch_dtype=torch.float16,  
    device_map="auto",  # 자동으로 GPU 설정
)

# 텍스트 생성 예시
text = "This is a test document."

# 텍스트 생성 실행
generated_text = pipeline(text, max_length=50, num_return_sequences=1)

# 출력 결과
print(generated_text)


# In[106]:


from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

hf = HuggingFacePipeline(pipeline=pipeline)


# In[107]:


from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import PromptTemplate

promptTemplate_fstring = """
You are a customer service assistant, tasked with providing clear and concise answers based on the given context. 
Context:
{context}
Question:
{query}
Answer:
"""


# In[108]:


from langchain_core.prompts import PromptTemplate

template = """Question: {question}
Answer:"""

prompt = PromptTemplate.from_template(template)
print(prompt)
chain = prompt | hf

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))


# In[ ]:





# In[3]:


# def run_vector_search(query: str):
#     # print(query)
#     query_vector = db.similarity_search_with_score(query, k=5)
#     # print(query_vector)
#     # document, score = query_vector
    
#     # return query_vector.metadata['source'], document.page_content, score
#     return query_vector

def run_vector_search(query: str, k = 3) -> str:
    """Performs a vector search and returns results in a structured format."""
    query_vector = db.similarity_search_with_score(query, k=k)

    # Extract and structure the search results
    results = []
    for document, score in query_vector:
        # Extract metadata and page content
        metadata = document.metadata
        page_content = {k.strip(): v.strip() for k, v in [line.split(':', 1) for line in document.page_content.split('\n') if ':' in line]}
        
        # Combine all data into a single dictionary
        combined_result = {
            "source": metadata.get("source", "Unknown Source"),
            **page_content,
            "score": score
        }
        results.append(combined_result)

    # Convert to a DataFrame
    df_results = pd.DataFrame(results)
    return df_results


# In[ ]:





# In[5]:


def init_llm():
    """Initializes the LLM by reading the CSV file, loading the embeddings model, and vectorizing the documents.

    Returns:
        Chroma: A Chroma object that contains the vectorized documents.
    """
    # Replace 'read_csv' with the appropriate CSV reading logic
    data = read_csv(file_path='data/amazon_rag.csv', source_column="product_name")
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    # model_name = 'jhgan/ko-sroberta-nli'
    embedding_function = load_embeddings_model(model_name)
    db = vectorize_documents(data, embedding_function)
    return db


# In[ ]:





# In[7]:


# Assuming `db` is created in the `init_llm()` function, you can call `init_llm()` to get `db`
def run_vector_search(query, k=3):
    """Performs a vector search and returns results in a structured format."""
    
    # Initialize the LLM to get the vector database (db)
    db = init_llm()  # Make sure `init_llm` is called to initialize `db`
    
    # Vector search using `db`
    results = db.similarity_search_with_score(query, k=k)
    
    # Convert results to a pandas DataFrame (optional)
    results_df = pd.DataFrame(results, columns=["question", "score"])
    
    return results_df


# In[12]:


import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Hugging Face API 토큰 환경 변수 설정
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # 본인의 토큰으로 변경

# 모델 이름 지정
model_name = 'sentence-transformers/all-mpnet-base-v2'

# Hugging Face 모델 로드
def load_embeddings_model(model_name: str) -> HuggingFaceEmbeddings:
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_function

# 임베딩 모델 로드
embeddings = load_embeddings_model(model_name)

# 텍스트 예시
text = "This is a test document."
embedding = embeddings.embed_query(text)
print(embedding)


# In[ ]:





# In[ ]:





# In[16]:


import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Hugging Face API 토큰 환경 변수 설정
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # 본인의 토큰으로 변경

# 모델 이름 지정
model_name = 'sentence-transformers/all-mpnet-base-v2'

# Hugging Face 모델 로드
def load_embeddings_model(model_name: str) -> HuggingFaceEmbeddings:
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_function

# 임베딩 모델 로드
embeddings = load_embeddings_model(model_name)

# 텍스트 예시
text = "This is a test document."
embedding = embeddings.embed_query(text)
print(embedding)


# In[45]:


import os
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tqdm import tqdm  # tqdm을 임포트
from langchain.embeddings.huggingface import HuggingFaceEmbeddings  # HuggingFaceEmbeddings 사용
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.schema import Document
from transformers import pipeline  # Hugging Face 모델 로드용
from langchain.llms import HuggingFacePipeline

# Hugging Face API 토큰 환경 변수 설정
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # 본인의 토큰으로 변경

# 모델 이름 지정
model_name = 'sentence-transformers/all-mpnet-base-v2'

# Hugging Face 모델 로드 함수
def load_embeddings_model(model_name: str) -> HuggingFaceEmbeddings:
    """Loads a Hugging Face model and returns an Embeddings object."""
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_function

# CSV 파일을 읽어들이는 함수 정의
def read_csv(file_path: str, source_column: str = "question") -> pd.DataFrame:
    """Reads a CSV file and returns a DataFrame containing the relevant column."""
    data = pd.read_csv(file_path)
    return data

# 문서 벡터화 함수 정의
def vectorize_documents(data: pd.DataFrame, embedding_function) -> Chroma:
    """Vectorizes a list of Documents using the provided embedding function."""
    
    # 데이터를 문서 객체로 변환, 메타데이터는 'question'만 포함
    documents = [
        Document(page_content=row['question'], metadata={})
        for _, row in data.iterrows()
    ]
    
    # Chroma를 사용해 문서 벡터화
    db = Chroma.from_documents(documents, embedding_function)
    return db

# 필요한 함수 정의들 (예: vectorization, embeddings 로드 등)
def init_llm():
    """Initializes the LLM by reading the CSV file, loading the embeddings model, and vectorizing the documents."""
    data = read_csv(file_path='data/amazon_rag.csv', source_column="question")  # CSV 파일 로드
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    embedding_function = load_embeddings_model(model_name)  # 임베딩 모델 로드
    db = vectorize_documents(data, embedding_function)  # 문서 벡터화
    return db

def run_vector_search(query, k=3):
    """Performs a vector search and returns results in a structured format."""
    db = init_llm()  # db 초기화 (벡터화된 문서 데이터베이스)
    results = db.similarity_search_with_score(query, k=k)
    results_df = pd.DataFrame(results, columns=["question", "score"])  # 결과를 DataFrame으로 변환
    return results_df

# Hugging Face 모델 로드 (hf는 이제 Hugging Face 모델을 나타냄)
hf_pipeline = pipeline("text-generation", model="gpt2", device=0, max_new_tokens=100, temperature=0.2)  # max_new_tokens 및 temperature 추가

# Hugging Face 모델을 HuggingFacePipeline로 래핑하여 LLMChain에서 사용할 수 있게 함
hf = HuggingFacePipeline(pipeline=hf_pipeline)

# Query definition
query = "suggest cool iPhone USB charger and adapter"

# Perform vector search
doc_context = run_vector_search(query)

# Extract relevant columns (유니크한 'question'만 사용)
doc = doc_context[['question']].drop_duplicates()  # 중복 제거

# Convert context to string (using ' '.join() to join rows into one long string)
context = ' '.join([str(doc_content.page_content) for doc_content in doc['question']])

# Define the prompt template
promptTemplate_fstring = """
You are a customer service assistant, tasked with providing clear and concise answers based on the given context. 
Context:
{context}
Question:
{query}
Answer:
"""

# Initialize the prompt
prompt = PromptTemplate(
    template=promptTemplate_fstring,
)

# Initialize the LLM chain with your model (`hf` refers to your Hugging Face model)
chain = LLMChain(prompt=prompt, llm=hf)

# Create a tqdm progress bar for the LLMChain execution
with tqdm(total=100, desc="Running LLMChain", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}] [Remaining: {remaining}]") as pbar:
    # Run the chain and get the response
    response = chain.run({"query": query, "context": context})
    
    # Update progress bar to show completion
    pbar.update(100)

# Print the response
print(response)


# In[50]:


import os
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from tqdm import tqdm  # tqdm을 임포트
from langchain.embeddings.huggingface import HuggingFaceEmbeddings  # HuggingFaceEmbeddings 사용
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.schema import Document
from transformers import pipeline  # Hugging Face 모델 로드용
from langchain.llms import HuggingFacePipeline

# Hugging Face API 토큰 환경 변수 설정
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # 본인의 토큰으로 변경

# 모델 이름 지정
model_name = 'sentence-transformers/all-mpnet-base-v2'

# Hugging Face 모델 로드 함수
def load_embeddings_model(model_name: str) -> HuggingFaceEmbeddings:
    """Loads a Hugging Face model and returns an Embeddings object."""
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_function

# CSV 파일을 읽어들이는 함수 정의
def read_csv(file_path: str, source_column: str = "question") -> pd.DataFrame:
    """Reads a CSV file and returns a DataFrame containing the relevant column."""
    data = pd.read_csv(file_path)
    return data

# 문서 벡터화 함수 정의
def vectorize_documents(data: pd.DataFrame, embedding_function) -> Chroma:
    """Vectorizes a list of Documents using the provided embedding function."""
    
    # 데이터를 문서 객체로 변환, 메타데이터는 'question'만 포함
    documents = [
        Document(page_content=row['question'], metadata={})
        for _, row in data.iterrows()
    ]
    
    # Chroma를 사용해 문서 벡터화
    db = Chroma.from_documents(documents, embedding_function)
    return db

# 필요한 함수 정의들 (예: vectorization, embeddings 로드 등)
def init_llm():
    """Initializes the LLM by reading the CSV file, loading the embeddings model, and vectorizing the documents."""
    data = read_csv(file_path='data/amazon_rag.csv', source_column="question")  # CSV 파일 로드
    embedding_function = load_embeddings_model(model_name)  # 임베딩 모델 로드
    db = vectorize_documents(data, embedding_function)  # 문서 벡터화
    return db

def run_vector_search(query, k=3):
    """Performs a vector search and returns results in a structured format."""
    db = init_llm()  # db 초기화 (벡터화된 문서 데이터베이스)
    results = db.similarity_search_with_score(query, k=k)
    
    # results의 실제 구조를 확인하여 DataFrame 생성
    print("Results:", results)  # 결과 확인 (디버깅용)

    # 결과가 2개의 열만 있을 수 있기 때문에, 'question', 'score'만 사용
    results_df = pd.DataFrame(results, columns=["question", "score"])  # 필요한 열만 선택
    return results_df

# Hugging Face 모델 로드 (hf는 이제 Hugging Face 모델을 나타냄)
hf_pipeline = pipeline("text-generation", model="gpt2", device=0, max_new_tokens=100, temperature=0.2)  # max_new_tokens 및 temperature 추가

# Hugging Face 모델을 HuggingFacePipeline로 래핑하여 LLMChain에서 사용할 수 있게 함
hf = HuggingFacePipeline(pipeline=hf_pipeline)

# Query definition
query = "suggest cool iPhone USB charger and adapter"

# Perform vector search
doc_context = run_vector_search(query)

# Extract relevant columns (유니크한 'question'만 사용)
doc = doc_context[['question']].drop_duplicates()  # 중복 제거

# Convert context to string (using ' '.join() to join rows into one long string)
context = ' '.join([str(row['question']) for _, row in doc.iterrows()])

# Define the prompt template
promptTemplate_fstring = """
You are a customer service assistant, tasked with providing clear and concise answers based on the given context. 
Context:
{context}
Question:
{query}
Answer:
"""

# Initialize the prompt
prompt = PromptTemplate(template=promptTemplate_fstring)

# Initialize the LLM chain with your model (`hf` refers to your Hugging Face model)
chain = LLMChain(prompt=prompt, llm=hf)

# Create a tqdm progress bar for the LLMChain execution
with tqdm(total=100, desc="Running LLMChain", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}] [Remaining: {remaining}]") as pbar:
    # Run the chain and get the response
    response = chain.run({"query": query, "context": context})
    
    # Update progress bar to show completion
    pbar.update(100)

# Print the response
print(response)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




