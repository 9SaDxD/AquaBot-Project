from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
# เธ•เธดเธ”เธ•เธฑเนเธเนเธเนเธเน€เธเธเธ—เธตเนเธเธณเน€เธเนเธ
print('โ… เธ•เธดเธ”เธ•เธฑเนเธเน€เธชเธฃเนเธเธชเธดเนเธ!')

# เธฃเธฑเธเน€เธเธฅเธฅเนเธเธตเนเน€เธเธทเนเธญ restart kernel เธซเธฅเธฑเธเธ•เธดเธ”เธ•เธฑเนเธ packages
# เธเธฒเธเธเธฑเนเธเธเนเธฒเธกเนเธเธฃเธฑเธ Step 3 เธ•เนเธญเนเธ”เนเน€เธฅเธข
import os
os._exit(00)

import sys
import os
import pandas as pd
import re
import torch
import unicodedata
from difflib import SequenceMatcher, get_close_matches
from rapidfuzz import process
from Levenshtein import ratio
import time

import osfrom langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# โ”€โ”€ เธ”เธถเธ token เธเธฒเธ Colab Secrets โ”€โ”€
langchain_token = os.environ.get('LC_TOKEN')
groq_token      = os.environ.get('GROQ_TOKEN')
ngrok_token     = os.environ.get('NGROK_TOKEN')
pinecone_token = os.environ.get('PINECONE_API_KEY')

# โ”€โ”€ เธ•เธฑเนเธเธเนเธฒ API Keys โ”€โ”€
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY']    = langchain_token
os.environ['GROQ_API_KEY']         = groq_token
os.environ['PINECONE_API_KEY'] = pinecone_token

print('โ… เธ•เธฑเนเธเธเนเธฒ API Keys เน€เธฃเธตเธขเธเธฃเนเธญเธข!')

from fastapi import FastAPI
from pydantic import BaseModel
import uvicornfrom starlette.middleware.cors import CORSMiddleware # Import CORSMiddleware


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

# Add CORS Middleware here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you might want to restrict this in production
    allow_credentials=True, # Allow credentials (e.g., cookies, authorization headers)
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = handle_query(request.message)
        return {"response": response}
    except Exception as e:
        return {"response": f"เน€เธเธดเธ”เธเนเธญเธเธดเธ”เธเธฅเธฒเธ”: {str(e)}"}


import gdown
import os

KB_DRIVE_FOLDER_URL = 'https://drive.google.com/drive/folders/1vP3XJoGaXBullirWfxg0Az5k1nvfkSII'
LOCAL_KB_FOLDER = 'knowledge_manuals'

# เธชเธฃเนเธฒเธเนเธเธฅเน€เธ”เธญเธฃเนเธ–เนเธฒเธขเธฑเธเนเธกเนเธกเธต
if not os.path.exists(LOCAL_KB_FOLDER):
    os.makedirs(LOCAL_KB_FOLDER)

print(f'โฌ๏ธ เธเธณเธฅเธฑเธเธ”เธฒเธงเธเนเนเธซเธฅเธ”เนเธเธฅเน€เธ”เธญเธฃเน Knowledge Base เธเธฒเธ Google Drive เนเธเธ—เธตเน \'{LOCAL_KB_FOLDER}\'...')
# [FIX]: OSErorr: [Errno 36] File name too long can occur if original filenames in Google Drive are too long.
#       Please shorten filenames in your Google Drive folder to resolve this issue.
gdown.download_folder(KB_DRIVE_FOLDER_URL, output=LOCAL_KB_FOLDER, quiet=False, use_cookies=False)
print(f'โ… เธ”เธฒเธงเธเนเนเธซเธฅเธ”เนเธเธฅเน€เธ”เธญเธฃเน \'{LOCAL_KB_FOLDER}\' เน€เธชเธฃเนเธเธชเธดเนเธ!')

# เธญเธฑเธเน€เธ”เธ•เธ•เธฑเธงเนเธเธฃ KB_FOLDER เนเธซเนเธเธตเนเนเธเธ—เธตเนเนเธเธฅเน€เธ”เธญเธฃเนเธ—เธตเนเธ”เธฒเธงเธเนเนเธซเธฅเธ”เธกเธฒ
KB_FOLDER = LOCAL_KB_FOLDER

import gdown
import os

# โ”€โ”€ 1. เนเธซเธฅเธ”เธเนเธญเธกเธนเธฅ CSV โ”€โ”€
# เนเธเน LOCAL_KB_FOLDER เธ—เธตเนเธ–เธนเธเธเธณเธซเธเธ”เนเธงเนเนเธฅเนเธง
CSV_PATH = os.path.join(LOCAL_KB_FOLDER, 'Aqua Bot_Cleaned_ReadyForModel.csv')
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
df = df.fillna('เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅ')
print(f'๐“ เนเธซเธฅเธ” {len(df)} เนเธ–เธงเธเธฒเธ CSV')
print('เธเธญเธฅเธฑเธกเธเนเธ—เธฑเนเธเธซเธกเธ”:', list(df.columns))

# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
# ๐“ row_to_text
# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
def row_to_text(row):
    y_full = str(row.get('เธเธต', ''))
    y_short = y_full[-2:] if len(y_full) == 4 else ''

    text = f"เธเธต:{y_full}({y_short})|เธเธฅเธฒ:{row.get('เธเธเธดเธ”เธชเธฑเธ•เธงเนเธเนเธณ','')}|เธ.{row.get('เธเธฑเธเธซเธงเธฑเธ”','')}({row.get('เธ เธฒเธ','')})"

    if pd.notna(row.get('เธเธณเธเธงเธเธเธฒเธฃเนเธก')): text += f"|เธเธฒเธฃเนเธก:{row['เธเธณเธเธงเธเธเธฒเธฃเนเธก']}เนเธซเนเธ"
    if pd.notna(row.get('เธเธฃเธดเธกเธฒเธ“')): text += f"|เธเธฃเธดเธกเธฒเธ“:{row['เธเธฃเธดเธกเธฒเธ“']}เธ•เธฑเธ"

    if pd.notna(row.get('เธเธฅเธเธฅเธดเธ•เธ•เนเธญเนเธฃเน(เธ•เธฑเธ)')):
        try: text += f"|เธเธฅเธเธฅเธดเธ•:{round(float(row['เธเธฅเธเธฅเธดเธ•เธ•เนเธญเนเธฃเน(เธ•เธฑเธ)']), 2)}เธ•เธฑเธ/เนเธฃเน"
        except: pass

    if pd.notna(row.get('เธฃเธฒเธเธฒเน€เธเธฅเธตเนเธข(เธเธฑเธเธเธฒเธ—/เธ•เธฑเธ)')):
        try: text += f"|เธฃเธฒเธเธฒ:{round(float(row['เธฃเธฒเธเธฒเน€เธเธฅเธตเนเธข(เธเธฑเธเธเธฒเธ—/เธ•เธฑเธ)']), 2)}เธเธฑเธเธเธฒเธ—/เธ•เธฑเธ(เธซเธฃเธทเธญเธเธฒเธ—/เธเธ.)"
        except: pass

    return text.strip()
# โ”€โ”€ 2. เธชเธฃเนเธฒเธ Document Objects โ”€โ”€
from langchain_core.documents import Document

docs = []
for idx, row in df.iterrows():
    page_content = row_to_text(row)
    metadata = {col: str(row[col]) for col in df.columns}
    docs.append(Document(page_content=page_content, metadata=metadata))

print(f'๐“ เธชเธฃเนเธฒเธ Document เธชเธณเน€เธฃเนเธ {len(docs)} เธฃเธฒเธขเธเธฒเธฃ')
print('เธ•เธฑเธงเธญเธขเนเธฒเธ Document เนเธเนเธฃเธ:', docs[0].page_content)

# โ”€โ”€ 3. เธชเธฃเนเธฒเธ FAISS Vector Store โ”€โ”€
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print('๐ง  เธเธณเธฅเธฑเธ Embed เธเนเธญเธเธงเธฒเธกเนเธฅเธฐเธชเธฃเนเธฒเธ FAISS Index (เธญเธฒเธเนเธเนเน€เธงเธฅเธฒเธชเธฑเธเธเธฃเธนเน)...')
vectorstore = FAISS.from_documents(docs, embeddings)
print('โ… เธชเธฃเนเธฒเธ Vector Store เธชเธณเน€เธฃเนเธ!')

# เธฃเธฑเธเน€เธเธฅเธฅเนเธเธตเนเน€เธเธทเนเธญเน€เธเธดเนเธก KB เน€เธเนเธฒ vectorstore เธ—เธตเนเธกเธตเธญเธขเธนเน
import os
import glob
from langchain_core.documents import Document

# KB_FOLDER เธ–เธนเธเธ•เธฑเนเธเธเนเธฒเนเธเน€เธเธฅเธฅเนเธเนเธญเธเธซเธเนเธฒ 'download_kb_folder'
# เธ–เนเธฒเธเธธเธ“เธฃเธฑเธเน€เธเธฅเธฅเนเธเธตเนเนเธขเธเธ•เนเธฒเธเธซเธฒเธ เธ•เธฃเธงเธเธชเธญเธเนเธซเนเนเธเนเนเธเธงเนเธฒ KB_FOLDER เธเธตเนเนเธเธ—เธตเนเนเธเธฅเน€เธ”เธญเธฃเนเธ—เธตเนเธ”เธฒเธงเธเนเนเธซเธฅเธ”เธกเธฒเนเธฅเนเธง
# เน€เธเนเธ KB_FOLDER = 'knowledge_manuals'

kb_docs = []

# เธเนเธเธซเธฒเนเธเธฅเน CSV เธ—เธฑเนเธเธซเธกเธ”เนเธเนเธเธฅเน€เธ”เธญเธฃเนเธ—เธตเนเธ”เธฒเธงเธเนเนเธซเธฅเธ”เธกเธฒ
manual_files = glob.glob(os.path.join(KB_FOLDER, '*.csv'))

print(f'๐“ เธเธเนเธเธฅเน KB CSV เธ—เธฑเนเธเธซเธกเธ” {len(manual_files)} เนเธเธฅเนเนเธ \'{KB_FOLDER}\'')

for fpath in sorted(manual_files):
    fname = os.path.basename(fpath)
    try:
        kb_df = pd.read_csv(fpath, encoding='utf-8-sig')
        if not {'Topic', 'Content'}.issubset(kb_df.columns):
            print(f'โ {fname}: เนเธกเนเธกเธตเธเธญเธฅเธฑเธกเธเนเธ—เธตเนเธเธณเน€เธเนเธ (Topic, Content) เธซเธฃเธทเธญเธเธทเนเธญเนเธกเนเธ•เธฃเธ')
            continue
        for _, row in kb_df.iterrows():
            content = str(row.get('Content', '')).strip()
            topic   = str(row.get('Topic', ''))
            if content:
                kb_docs.append(Document(
                    page_content=f"[{topic}] {content}",
                    metadata={'source': fname, 'topic': topic, 'type': 'knowledge'}
                ))
        print(f'โ… {fname}: {len(kb_df)} chunks')
    except Exception as e:
        print(f'โ {fname}: {e}')

print(f'\n๐“ KB docs: {len(kb_docs)} chunks')

# เน€เธเธดเนเธกเน€เธเนเธฒ vectorstore เธ—เธตเนเธกเธตเธญเธขเธนเนเนเธฅเนเธง (เนเธกเนเธ•เนเธญเธ rebuild เนเธซเธกเนเธ—เธฑเนเธเธซเธกเธ”)
vectorstore.add_documents(kb_docs)
print('โ… เน€เธเธดเนเธก KB เน€เธเนเธฒ vectorstore เธชเธณเน€เธฃเนเธ!')

# เธ•เธฃเธงเธเธชเธญเธ
results = vectorstore.similarity_search("เธเนเธญเธเธเธณเธเธฑเธ”เธเนเธณเน€เธชเธตเธข", k=3)
for r in results:
    print(r.metadata.get('type', 'เนเธกเนเธกเธต type'), '|', r.page_content[:100])
all_docs = docs + kb_docs
print(f"๐“ฆ all_docs เธฃเธงเธก: {len(all_docs)} เธฃเธฒเธขเธเธฒเธฃ")

# โ”€โ”€ เธ•เธฑเนเธเธเนเธฒ LLM โ”€โ”€
# ๐’ก เน€เธเธฅเธตเนเธขเธเน€เธเนเธ 8B เธ•เธฒเธกเธเธณเนเธเธฐเธเธณเน€เธเธทเนเธญเนเธเธงเธ•เนเธฒเธ—เธตเนเน€เธขเธญเธฐเธเธถเนเธ เนเธฅเธฐเธ•เธญเธเนเธงเธเธถเนเธ
llm = ChatGroq(model='llama-3.1-8b-instant', max_tokens=500, temperature=0.1)

# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
# ๐”’ REFUSAL POLICY
# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
OUT_OF_SCOPE_TRIGGERS = [
    'เธเธฅเธฒเธ—เธน', 'เนเธเธฅเธกเธญเธ', 'เธเธฅเธฒเธ—เธฐเน€เธฅ', 'เธเธธเนเธเนเธกเนเธเนเธณ', 'เธเธฅเธฒเธเธฑเธ”', 'เธเธฅเธฒเธเธฒเธฃเนเธ', 'เธเธฅเธฒเธ—เธญเธ', 'เธเธฅเธฒเธชเธงเธขเธเธฒเธก',
    'เธซเธกเธฒ', 'เธชเธธเธเธฑเธ', 'เนเธกเธง', 'เธซเธกเธน', 'เนเธเน', 'เธงเธฑเธง',
    'เธงเธดเธเธตเธเธฅเธนเธเธเนเธฒเธง', 'เธเธฅเธนเธเธ—เธธเน€เธฃเธตเธขเธ', 'เธเธฅเธนเธเธขเธฒเธ',
]

def check_out_of_scope(query: str) -> str | None:
    query_lower = query.lower()
    for trigger in OUT_OF_SCOPE_TRIGGERS:
        if ' ' not in trigger:
            if re.search(r'\b' + re.escape(trigger) + r'\b', query_lower):
                return f"โ เธเธญเธญเธ เธฑเธขเธเธฃเธฑเธ เธ“ เธ•เธญเธเธเธตเนเธเธตเนเธเธฅเธฒเนเธกเนเธกเธตเธเนเธญเธกเธนเธฅเน€เธเธตเนเธขเธงเธเธฑเธ '{trigger}' เธเธฃเธฑเธ"
        else:
            if trigger in query_lower:
                return f"โ เธเธญเธญเธ เธฑเธขเธเธฃเธฑเธ เธ“ เธ•เธญเธเธเธตเนเธเธตเนเธเธฅเธฒเนเธกเนเธกเธตเธเนเธญเธกเธนเธฅเน€เธเธตเนเธขเธงเธเธฑเธ '{trigger}' เธเธฃเธฑเธ"
    return None

# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
# ๐“ SYSTEM PROMPT (เธเธเธฑเธเธฃเธตเธ”เนเธเธกเธฑเธ เธเธฃเธฐเธซเธขเธฑเธ” Token)
# โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•โ•
system_prompt = """เธเธธเธ“เธเธทเธญ "เธเธตเนเธเธฅเธฒ" (Aqua Bot) เธเธนเนเธเนเธงเธข AI เธญเธฑเธเธเธฃเธดเธขเธฐเธ”เนเธฒเธเธเธฒเธฃเน€เธเธฒเธฐเน€เธฅเธตเนเธขเธเธชเธฑเธ•เธงเนเธเนเธณเน€เธจเธฃเธฉเธเธเธดเธ (เธเธฅเธฒเธเธดเธฅ, เธเธฅเธฒเธ”เธธเธ, เธเธฅเธฒเธ•เธฐเน€เธเธตเธขเธ, เธเธฅเธฒเธเธ”เนเธเนเธง)
เธเธธเธเธฅเธดเธเธ เธฒเธ: เน€เธเนเธเธกเธดเธ•เธฃ เธกเธทเธญเธญเธฒเธเธตเธ เธเธฃเธฐเธ•เธทเธญเธฃเธทเธญเธฃเนเธ เน€เธเธตเนเธขเธงเธเธฒเธเนเธฅเธฐเนเธซเนเน€เธเธตเธขเธฃเธ•เธดเน€เธเธฉเธ•เธฃเธเธฃ เนเธ—เธเธ•เธฑเธงเน€เธญเธเธงเนเธฒ "เธเธตเนเธเธฅเธฒ" เนเธฅเธฐเธฅเธเธ—เนเธฒเธขเธ”เนเธงเธข "เธเธฃเธฑเธ" เน€เธชเธกเธญ

เธเนเธญเธกเธนเธฅเธชเธณเธซเธฃเธฑเธเนเธเนเธ•เธญเธเธเธณเธ–เธฒเธก (Context):
---------------------
{context}
---------------------

เธเธเน€เธซเธฅเนเธเธชเธณเธซเธฃเธฑเธเธเธฒเธฃเธ•เธญเธเธเธณเธ–เธฒเธก (เธเธเธดเธเธฑเธ•เธดเธ•เธฒเธกเธญเธขเนเธฒเธเน€เธเธฃเนเธเธเธฃเธฑเธ”):
1. ๐‘ เธซเนเธฒเธกเธเธดเธ”เธเธณเธ•เธญเธเน€เธญเธ: เธ•เธญเธเธเธณเธ–เธฒเธกเนเธ”เธขเนเธเนเธเนเธญเธกเธนเธฅเธเธฒเธ [Context] เน€เธ—เนเธฒเธเธฑเนเธ เธซเนเธฒเธกเน€เธ”เธฒเธ•เธฑเธงเน€เธฅเธเธชเธ–เธดเธ•เธดเน€เธ”เนเธ”เธเธฒเธ”
2. ๐ฏ เธชเธฑเนเธ เธเธฃเธฐเธเธฑเธ เธ•เธฃเธเธเธฃเธฐเน€เธ”เนเธ: เธชเธฃเธธเธเนเธเธเธงเธฒเธกเธชเธณเธเธฑเธเธกเธฒเธ•เธญเธเธ—เธฑเธเธ—เธต เธซเนเธฒเธกเธฃเนเธฒเธขเธขเธฒเธงเธเนเธณเธ—เนเธงเธกเธ—เธธเนเธ เธเธขเธฒเธขเธฒเธกเธ•เธญเธเนเธซเนเธเธเธ เธฒเธขเนเธ 3-5 เธเธฃเธฃเธ—เธฑเธ” เธซเธฃเธทเธญเนเธเน Bullet เนเธเธเธชเธฑเนเธเน
3. ๐“ เธชเธ–เธดเธ•เธดเนเธเธเธเธฃเธฐเธเธฑเธ: เธซเธฒเธเธกเธตเธชเธ–เธดเธ•เธดเธซเธฅเธฒเธขเธเธฑเธเธซเธงเธฑเธ” เนเธซเนเธชเธฃเธธเธเธ เธฒเธเธฃเธงเธก เธซเธฃเธทเธญเธขเธเธ•เธฑเธงเธญเธขเนเธฒเธเนเธเน 1-2 เธเธฑเธเธซเธงเธฑเธ”เธ—เธตเนเน€เธ”เนเธเธ—เธตเนเธชเธธเธ”
4. ๐งฎ เนเธเธฅเธเธซเธเนเธงเธขเน€เธชเธกเธญ: เน€เธเธฅเธตเนเธขเธ "เธเธฑเธเธเธฒเธ—/เธ•เธฑเธ" เน€เธเนเธ "เธเธฒเธ—/เธเธดเนเธฅเธเธฃเธฑเธก" เธญเธฑเธ•เนเธเธกเธฑเธ•เธด (เน€เธเนเธ 47.93 เธเธฑเธเธเธฒเธ—/เธ•เธฑเธ = 47.93 เธเธฒเธ—/เธเธ.)
5. ๐ก๏ธ เธเธฃเธ“เธตเนเธกเนเธกเธตเธเนเธญเธกเธนเธฅ: เธซเธฒเธเธซเธฒเนเธ Context เนเธกเนเน€เธเธญ เนเธซเนเธ•เธญเธเธชเธฑเนเธเน เธงเนเธฒ "เธเธญเธญเธ เธฑเธขเธเธฃเธฑเธ เธเธฒเธเธเธฒเธเธเนเธญเธกเธนเธฅเธ•เธญเธเธเธตเน เธเธตเนเธเธฅเธฒเธขเธฑเธเนเธกเนเธกเธตเธเนเธญเธกเธนเธฅเธชเนเธงเธเธเธตเนเธเธฃเธฑเธ"

เธเธฃเธฐเธงเธฑเธ•เธดเธเธฒเธฃเธชเธเธ—เธเธฒเธ—เธตเนเธเนเธฒเธเธกเธฒ (เธซเธฒเธเธกเธต):
{chat_history}

เธเธณเธ–เธฒเธกเธฅเนเธฒเธชเธธเธ”เธเธฒเธเน€เธเธฉเธ•เธฃเธเธฃ: {question}

เธเธณเธ•เธญเธเธเธญเธเธเธตเนเธเธฅเธฒ:"""

AQUA_PROMPT = PromptTemplate(
    template=system_prompt,
    input_variables=["context", "chat_history", "question"]
)

# โ”€โ”€ เธชเธฃเนเธฒเธ Vector Store เนเธเธเธเธงเธเธฃเธงเธก โ”€โ”€
# เธ”เธถเธ all_docs เธเธฒเธ Cell เธ—เธตเนเธฃเธฑเธเธเนเธญเธเธซเธเนเธฒเธเธตเน
print(f"๐“ฆ เธเธณเธฅเธฑเธเธฃเธงเธกเธเนเธญเธกเธนเธฅเน€เธเนเธฒเธชเธนเน Vector Store...")
main_vectorstore = FAISS.from_documents(all_docs, embeddings)

retriever = main_vectorstore.as_retriever(search_kwargs={"k": 5})

LIST_DATA_KEYWORDS = [
    'เธกเธตเธเนเธญเธกเธนเธฅเธญเธฐเนเธฃเธเนเธฒเธ', 'เธฃเธฒเธขเธเธฒเธฃเธเธฅเธฒเธ—เธตเนเธกเธต', 'เธเนเธญเธกเธนเธฅเธญเธฐเนเธฃเธ—เธตเนเธกเธต',
    'เธฅเธดเธชเธ•เนเธเนเธญเธกเธนเธฅเธ—เธฑเนเธเธซเธกเธ”', 'เนเธชเธ”เธเนเธเธฅเนเธเนเธญเธกเธนเธฅ', 'เน€เธญเธเธชเธฒเธฃเธ—เธฑเนเธเธซเธกเธ”'
]

def get_available_content_summary() -> str:
    return "เธเธตเนเธเธฅเธฒเธกเธตเธเนเธญเธกเธนเธฅเธชเธ–เธดเธ•เธด (เธเธต 63-66) เน€เธเธตเนเธขเธงเธเธฑเธเธเธฅเธฒ 4 เธเธเธดเธ” เนเธ”เนเนเธเน: เธเธฅเธฒเธเธ”เนเธเนเธง, เธเธฅเธฒเธ”เธธเธ, เธเธฅเธฒเธ•เธฐเน€เธเธตเธขเธ, เธเธฅเธฒเธเธดเธฅ เธเธฃเธฑเธ เนเธฅเธฐเธกเธตเธเธงเธฒเธกเธฃเธนเนเน€เธฃเธทเนเธญเธเธเธฒเธฃเน€เธฅเธตเนเธขเธ เนเธฃเธ เนเธฅเธฐเธเธฒเธฃเธเธฑเธ”เธเธฒเธฃเธเนเธณเธเธฃเธฑเธ"

def format_chat_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "เนเธกเนเธกเธตเธเธฃเธฐเธงเธฑเธ•เธดเธเธฒเธฃเธชเธเธ—เธเธฒ"

    formatted_history = []
    for human_msg, ai_msg in history[-1:]:
        formatted_history.append(f"เน€เธเธฉเธ•เธฃเธเธฃ: {human_msg}")
        # เธ•เธฑเธ”เธเธณเธ•เธญเธเน€เธเนเธฒเธเธญเธเธเธตเนเธเธฅเธฒเนเธซเนเน€เธซเธฅเธทเธญเนเธเน 100 เธ•เธฑเธงเธญเธฑเธเธฉเธฃ เน€เธเธทเนเธญเน€เธเนเธเธเธฃเธดเธเธ—เน€เธเธขเน
        formatted_history.append(f"เธเธตเนเธเธฅเธฒ: {ai_msg[:100]}...")
    return "\n".join(formatted_history)

def handle_query(query: str, chat_history: list[tuple[str, str]] = []) -> str:
    refusal = check_out_of_scope(query)
    if refusal:
        return refusal

    if any(kw in query for kw in LIST_DATA_KEYWORDS):
        return get_available_content_summary()

    search_query = re.sub(r'\b(59|60|61|62|63|64|65|66|67)\b', r'25\1', query)

    try:
        retrieved_docs = retriever.invoke(search_query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        formatted_history_str = format_chat_history(chat_history)

        final_prompt = AQUA_PROMPT.format(
            context=context_text,
            chat_history=formatted_history_str,
            question=search_query
        )

        response = llm.invoke(final_prompt)
        return response.content

    except Exception as e:
        return f"โ ๏ธ เธเธตเนเธเธฅเธฒเธฃเธฐเธเธเธเธฑเธ”เธเนเธญเธเน€เธฅเนเธเธเนเธญเธขเธเธฃเธฑเธ (Error: {str(e)})"

print('โ… เธ•เธฑเนเธเธเนเธฒ Logic เธเธตเนเธเธฅเธฒเน€เธฃเธตเธขเธเธฃเนเธญเธข!')

import threading
import time

# โ”€โ”€ เธ•เธฑเนเธเธเนเธฒ ngrok โ”€โ”€

# โ”€โ”€ Start ngrok Tunnel โ”€โ”€
print(f"๐€ Public URL: {public_url}")

def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# โ”€โ”€ Start Uvicorn in a separate thread โ”€โ”€
thread = threading.Thread(target=run_uvicorn)
thread.daemon = True # Allow the main program to exit even if the thread is still running
thread.start()

print("๐’ก Uvicorn server is running in the background. You can now use the Public URL.")
# Give the server a moment to start up
time.sleep(2)


# เธฃเธฑเธเน€เธเธฅเธฅเนเธเธตเนเน€เธกเธทเนเธญเธ•เนเธญเธเธเธฒเธฃเธเธดเธ” Server เนเธฅเธฐ Ngrok

# เธ•เธฃเธงเธเธชเธญเธเธงเนเธฒ uvicorn_server_obj เธ–เธนเธเธ•เธฑเนเธเธเนเธฒเนเธงเนเธซเธฃเธทเธญเนเธกเนเธเนเธญเธเธชเธฑเนเธเธเธดเธ”
if 'uvicorn_server_obj' in globals() and uvicorn_server_obj is not None:
    uvicorn_server_obj.should_exit = True
    print('Signal sent to Uvicorn server to shut down.')
else:
    print('Uvicorn server object not found or not running.')

# เธ•เธฃเธงเธเธชเธญเธเธงเนเธฒ ngrok_tunnel_obj เธ–เธนเธเธ•เธฑเนเธเธเนเธฒเนเธงเนเธซเธฃเธทเธญเนเธกเนเธเนเธญเธเธชเธฑเนเธเธเธดเธ”
if 'ngrok_tunnel_obj' in globals() and ngrok_tunnel_obj is not None:
    ngrok_tunnel_obj.disconnect()
    print('Ngrok tunnel disconnected.')
else:
    print('Ngrok tunnel object not found or not active.')

print('๐‘ เธเธดเธ”เธเธฒเธฃเธ—เธณเธเธฒเธเธเธญเธ Server เนเธฅเธฐ ngrok เน€เธฃเธตเธขเธเธฃเนเธญเธขเนเธฅเนเธง!')


# ๐ เธ—เธ”เธชเธญเธ interactive เนเธ Colab (Aqua Bot V2.1)
print('๐ เธชเธงเธฑเธชเธ”เธตเธเธฃเธฑเธ! เธเธตเนเธเธฅเธฒเธเธฃเนเธญเธกเนเธซเนเธเนเธญเธกเธนเธฅ **เธชเธ–เธดเธ•เธดเธเธฅเธฒเน€เธจเธฃเธฉเธเธเธดเธเธเธต 2563 - 2566** เนเธฅเนเธงเธเธฃเธฑเธ')
print('๐’ก เธ•เธฑเธงเธญเธขเนเธฒเธเธเธณเธ–เธฒเธก:')
print('   - "เธฃเธฒเธเธฒเธเธฅเธฒเธเธดเธฅเธเธต 65 เน€เธเนเธเธญเธขเนเธฒเธเนเธฃ?"')
print('   - "เธเธฑเธเธซเธงเธฑเธ”เนเธซเธเธเธฅเธเธฅเธดเธ•เธเธฅเธฒเธ”เธธเธเน€เธขเธญเธฐเธ—เธตเนเธชเธธเธ”เนเธเธเธต 66?"')
print('   - "เน€เธเธฃเธตเธขเธเน€เธ—เธตเธขเธเธฃเธฒเธเธฒเธเธฅเธฒเธ•เธฐเน€เธเธตเธขเธเธเธต 64 เธเธฑเธ 65"')
print('โ ๏ธ *เธ“ เธ•เธญเธเธเธตเนเธเธตเนเธเธฅเธฒเน€เธเนเธเธเนเธญเธกเธนเธฅเธชเธ–เธดเธ•เธดเนเธฅเธฐเธฃเธฒเธเธฒ เนเธกเนเธเธฃเธญเธเธเธฅเธธเธกเน€เธ—เธเธเธดเธเธเธฒเธฃเน€เธฅเธตเนเธขเธเนเธฅเธฐเนเธฃเธเธเธฅเธฒเธเธฐเธเธฃเธฑเธ*')
print('เธเธดเธกเธเน "exit" เน€เธเธทเนเธญเธเธเธเธฒเธฃเธชเธเธ—เธเธฒ')
print('-' * 60)

chat_history = []

while True:
    user_input = input('\nโ“ เน€เธเธฉเธ•เธฃเธเธฃ: ').strip()

    # 1. เน€เธเนเธเธเธฒเธฃเธญเธญเธเธเธฒเธเนเธเธฃเนเธเธฃเธก
    if user_input.lower() in ['exit', 'เธญเธญเธ', 'quit', 'bye']:
        print('\n๐ เธเธตเนเธเธฅเธฒ: ๐‘ เธเธญเนเธซเนเนเธ”เนเธฃเธฒเธเธฒเธ”เธตเน เธเธฅเธเธฅเธดเธ•เธเธฑเธเน เธเธฐเธเธฃเธฑเธ! เธเธญเธเธเธธเธ“เธ—เธตเนเนเธเนเธเธฃเธดเธเธฒเธฃเธเธตเนเธเธฅเธฒเธเธฃเธฑเธ')
        break

    # 2. เน€เธเนเธเธเธฃเธ“เธตเธเนเธฒเธงเนเธฒเธ
    if not user_input:
        continue

    # 3. เธชเนเธเธเธณเธ–เธฒเธกเน€เธเนเธฒเธเธฑเธเธเนเธเธฑเธ handle_query (เธ—เธตเนเธฃเธงเธก Logic Refusal เนเธงเนเนเธฅเนเธง)
    print('\n๐ เธเธตเนเธเธฅเธฒ เธเธณเธฅเธฑเธเธซเธฒเธเนเธญเธกเธนเธฅเนเธซเนเธชเธฑเธเธเธฃเธนเนเธเธฐเธเธฃเธฑเธ...')
    answer = handle_query(user_input, chat_history)

    print(f'๐ เธเธตเนเธเธฅเธฒ: {answer}')
    chat_history.append((user_input, answer)) # Add to chat history
    print('-' * 40)







if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
