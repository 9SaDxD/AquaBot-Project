import os
import sys
import glob
import time
import threading
import pandas as pd
import re
import torch
import unicodedata
from difflib import SequenceMatcher, get_close_matches
from rapidfuzz import process
from Levenshtein import ratio

# โหลด Environment Variables (ถ้ามีไฟล์ .env)
from dotenv import load_dotenv
load_dotenv()

# ตั้งค่า API Keys จาก Environment Variables (หรือ Colab Secrets)
langchain_token = os.environ.get('LC_TOKEN')
groq_token      = os.environ.get('GROQ_TOKEN')
ngrok_token     = os.environ.get('NGROK_TOKEN')
pinecone_token  = os.environ.get('PINECONE_API_KEY')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
if langchain_token:
    os.environ['LANGCHAIN_API_KEY'] = langchain_token
if groq_token:
    os.environ['GROQ_API_KEY'] = groq_token
if pinecone_token:
    os.environ['PINECONE_API_KEY'] = pinecone_token

print('✅ ตั้งค่า API Keys เรียบร้อย!')

# ──────────────────────────────────────────────
# 1. ดาวน์โหลด Knowledge Base จาก Google Drive
# ──────────────────────────────────────────────
import gdown

KB_DRIVE_FOLDER_URL = 'https://drive.google.com/drive/folders/1vP3XJoGaXBullirWfxg0Az5k1nvfkSII'
LOCAL_KB_FOLDER = 'knowledge_manuals'

if not os.path.exists(LOCAL_KB_FOLDER):
    os.makedirs(LOCAL_KB_FOLDER)

print(f'📥 กำลังดาวน์โหลดโฟลเดอร์ Knowledge Base จาก Google Drive ไปที่ \'{LOCAL_KB_FOLDER}\'...')
gdown.download_folder(KB_DRIVE_FOLDER_URL, output=LOCAL_KB_FOLDER, quiet=False, use_cookies=False)
print(f'✅ ดาวน์โหลดโฟลเดอร์ \'{LOCAL_KB_FOLDER}\' เสร็จสิ้น!')

KB_FOLDER = LOCAL_KB_FOLDER

# ──────────────────────────────────────────────
# 2. โหลดข้อมูล CSV และสร้าง Document / Vector Store
# ──────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

CSV_PATH = os.path.join(LOCAL_KB_FOLDER, 'Aqua Bot_Cleaned_ReadyForModel.csv')
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    df = df.fillna('ไม่มีข้อมูล')
    print(f'📊 โหลด {len(df)} แถวจาก CSV หลัก')
else:
    print(f'⚠️ไม่พบไฟล์ CSV หลักที่ {CSV_PATH}')
    df = pd.DataFrame()

def row_to_text(row):
    y_full = str(row.get('ปี', ''))
    y_short = y_full[-2:] if len(y_full) == 4 else ''

    text = f"ปี:{y_full}({y_short})|ปลา:{row.get('ชนิดสัตว์น้ำ','')}|จ.{row.get('จังหวัด','')}({row.get('ภาค','')})"

    if pd.notna(row.get('จำนวนฟาร์ม')): text += f"|ฟาร์ม:{row['จำนวนฟาร์ม']}แห่ง"
    if pd.notna(row.get('ปริมาณ')): text += f"|ปริมาณ:{row['ปริมาณ']}ตัน"

    if pd.notna(row.get('ผลผลิตต่อไร่(ตัน)')):
        try: text += f"|ผลผลิต:{round(float(row['ผลผลิตต่อไร่(ตัน)']), 2)}ตัน/ไร่"
        except: pass

    if pd.notna(row.get('ราคาเฉลี่ย(พันบาท/ตัน)')):
        try: text += f"|ราคา:{round(float(row['ราคาเฉลี่ย(พันบาท/ตัน)']), 2)}พันบาท/ตัน(หรือบาท/กก.)"
        except: pass

    return text.strip()

docs = []
for idx, row in df.iterrows():
    page_content = row_to_text(row)
    metadata = {col: str(row[col]) for col in df.columns}
    docs.append(Document(page_content=page_content, metadata=metadata))

print(f'📄 สร้าง Document หลักสำเร็จ {len(docs)} รายการ')

# โหลดไฟล์คู่มือ/KB เพิ่มเติมที่เป็น CSV ในโฟลเดอร์
kb_docs = []
manual_files = glob.glob(os.path.join(KB_FOLDER, '*.csv'))
print(f'📂 พบไฟล์ KB CSV ทั้งหมด {len(manual_files)} ไฟล์ใน \'{KB_FOLDER}\'')

for fpath in sorted(manual_files):
    fname = os.path.basename(fpath)
    if fname == 'Aqua Bot_Cleaned_ReadyForModel.csv':
        continue
    try:
        kb_df = pd.read_csv(fpath, encoding='utf-8-sig')
        if not {'Topic', 'Content'}.issubset(kb_df.columns):
            print(f'❌ {fname}: ไม่มีคอลัมน์ที่จำเป็น (Topic, Content) หรือชื่อไม่ตรง')
            continue
        for _, row in kb_df.iterrows():
            content = str(row.get('Content', '')).strip()
            topic   = str(row.get('Topic', ''))
            if content:
                kb_docs.append(Document(
                    page_content=f"[{topic}] {content}",
                    metadata={'source': fname, 'topic': topic, 'type': 'knowledge'}
                ))
        print(f'✅ {fname}: {len(kb_df)} chunks')
    except Exception as e:
        print(f'❌ {fname}: {e}')

all_docs = docs + kb_docs
print(f'📚 all_docs รวม: {len(all_docs)} รายการ')

# สร้าง Embeddings และ Vector Store ครั้งเดียวจบ
print('🧠 กำลัง Embed ข้อความและสร้าง FAISS Index (อาจใช้เวลาสักครู่)...')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
main_vectorstore = FAISS.from_documents(all_docs, embeddings)
retriever = main_vectorstore.as_retriever(search_kwargs={"k": 5})
print('✅ สร้าง Vector Store สำเร็จ!')

# ──────────────────────────────────────────────
# 3. ตั้งค่า LLM & Prompt
# ──────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

llm = ChatGroq(model='llama-3.1-8b-instant', max_tokens=500, temperature=0.1)

OUT_OF_SCOPE_TRIGGERS = [
    'ปลาทู', 'แซลมอน', 'ปลาทะเล', 'กุ้งแม่น้ำ', 'ปลากัด', 'ปลาคาร์พ', 'ปลาทอง', 'ปลาสวยงาม',
    'หมา', 'สุนัข', 'แมว', 'หมู', 'ไก่', 'วัว',
    'วิธีปลูกข้าว', 'ปลูกทุเรียน', 'ปลูกยาง',
]

def check_out_of_scope(query: str) -> str | None:
    query_lower = query.lower()
    for trigger in OUT_OF_SCOPE_TRIGGERS:
        if ' ' not in trigger:
            if re.search(r'\b' + re.escape(trigger) + r'\b', query_lower):
                return f"❌ ขออภัยครับ ณ ตอนนี้พี่ปลาไม่มีข้อมูลเกี่ยวกับ '{trigger}' ครับ"
        else:
            if trigger in query_lower:
                return f"❌ ขออภัยครับ ณ ตอนนี้พี่ปลาไม่มีข้อมูลเกี่ยวกับ '{trigger}' ครับ"
    return None

system_prompt = """คุณคือ "พี่ปลา" (Aqua Bot) ผู้ช่วย AI อัจฉริยะด้านการเพาะเลี้ยงสัตว์น้ำเศรษฐกิจ (ปลานิล, ปลาดุก, ปลาตะเพียน, ปลาฎแก้ว)
บุคลิกภาพ: เป็นมิตร มืออาชีพ กระตือรือร้น เชี่ยวชาญและให้เกียรติเกษตรกร แทนตัวเองว่า "พี่ปลา" และลงท้ายด้วย "ครับ" เสมอ

ข้อมูลสำหรับใช้ตอบคำถาม (Context):
---------------------
{context}
---------------------

กฎเหล็กสำหรับการตอบคำถาม (ปฏิบัติตามอย่างเคร่งครัด):
1. 🛑 ห้ามคิดคำตอบเอง: ตอบคำถามโดยใช้ข้อมูลจาก [Context] เท่านั้น ห้ามเดาตัวเลขสถิติเด็ดขาด
2. 🎯 สั้น กระชับ ตรงประเด็น: สรุปใจความสำคัญมาตอบทันที ห้ามร่ายยาวน้ำท่วมทุ่ง พยายามตอบให้จบภายใน 3-5 บรรทัด หรือใช้ Bullet แบบสั้นๆ
3. 📊 สถิติแบบกระชับ: หากมีสถิติหลายจังหวัด ให้สรุปภาพรวม หรือยกตัวอย่างแค่ 1-2 จังหวัดที่เด่นที่สุด
4. 🧮 แปลงหน่วยเสมอ: เปลี่ยน "พันบาท/ตัน" เป็น "บาท/กิโลกรัม" อัตโนมัติ (เช่น 47.93 พันบาท/ตัน = 47.93 บาท/กก.)
5. 🛡️ กรณีไม่มีข้อมูล: หากหาใน Context ไม่เจอ ให้ตอบสั้นๆ ว่า "ขออภัยครับ จากฐานข้อมูลตอนนี้ พี่ปลายยังไม่มีข้อมูลส่วนนี้ครับ"

ประวัติการสนทนาที่ผ่านมา (หากมี):
{chat_history}

คำถามล่าสุดจากเกษตรกร: {question}

คำตอบของพี่ปลา:"""

AQUA_PROMPT = PromptTemplate(
    template=system_prompt,
    input_variables=["context", "chat_history", "question"]
)

LIST_DATA_KEYWORDS = [
    'มีข้อมูลอะไรบ้าง', 'รายการปลาที่มี', 'ข้อมูลอะไรที่มี',
    'ลิสต์ข้อมูลทั้งหมด', 'แสดงไฟล์ข้อมูล', 'เอกสารทั้งหมด'
]

def get_available_content_summary() -> str:
    return "พี่ปลามีข้อมูลสถิติ (ปี 63-66) เกี่ยวกับปลา 4 ชนิด ได้แก่: ปลากดแก้ว, ปลาดุก, ปลาตะเพียน, ปลานิล ครับ และมีความรู้เรื่องการเลี้ยง โรค และการจัดการน้ำครับ"

def format_chat_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "ไม่มีประวัติการสนทนา"

    formatted_history = []
    for human_msg, ai_msg in history[-1:]:
        formatted_history.append(f"เกษตรกร: {human_msg}")
        formatted_history.append(f"พี่ปลา: {ai_msg[:100]}...")
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
        return f"⚠️ พี่ปลาระบบขัดข้องเล็กน้อยครับ (Error: {str(e)})"

print('✅ ตั้งค่า Logic พี่ปลาเรียบร้อย!')

# ──────────────────────────────────────────────
# 4. ตั้งค่า FastAPI & CORS
# ──────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = handle_query(request.message)
        return {"response": response}
    except Exception as e:
        return {"response": f"เกิดข้อผิดพลาด: {str(e)}"}

# ──────────────────────────────────────────────
# 5. รัน Uvicorn Server (Background Thread สำหรับ Colab)
# ──────────────────────────────────────────────
def run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=run_uvicorn)
thread.daemon = True
thread.start()

print("💡 Uvicorn server is running in the background.")
time.sleep(2)

# ──────────────────────────────────────────────
# 6. Interactive Chat สำหรับทดสอบใน Colab
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print('🐟 สวัสดีครับ! พี่ปลาพร้อมให้ข้อมูล **สถิติปลาเศรษฐกิจปี 2563 - 2566** แล้วครับ')
    print('💡 ตัวอย่างคำถาม:')
    print('   - "ราคาปลานิลปี 65 เป็นอย่างไร?"')
    print('   - "จังหวัดไหนผลผลิตปลาดุกเยอะที่สุดในปี 66?"')
    print('   - "เปรียบเทียบราคาปลาตะเพียนปี 64 กับ 65"')
    print('⚠️ *ณ ตอนนี้พี่ปลาเน้นข้อมูลสถิติและราคา ไม่ครอบคลุมเทคนิคการเลี้ยงและโรคปลาะครับ*')
    print('พิมพ์ "exit" เพื่อจบการสนทนา')
    print('-' * 60)

    chat_history = []

    # หากรันบน Script ทั่วไปที่ไม่ใช่ Interactive Input ให้ข้าม loop นี้ได้
    try:
        while True:
            user_input = input('\n❓ เกษตรกร: ').strip()
            if user_input.lower() in ['exit', 'ออก', 'quit', 'bye']:
                print('\n🐟 พี่ปลา:  👋 ขอให้ได้ราคาดีๆ ผลผลิตปังๆ นะครับ! ขอบคุณที่ใช้บริการพี่ปลาครับ')
                break
            if not user_input:
                continue

            print('\n🐟 พี่ปลา กำลังหาข้อมูลให้สักครู่นะครับ...')
            answer = handle_query(user_input, chat_history)
            print(f'🐟 พี่ปลา: {answer}')
            chat_history.append((user_input, answer))
            print('-' * 40)
    except (EOFError, KeyboardInterrupt):
        print("\nServer / Interactive session ended.")
