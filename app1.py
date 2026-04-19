import streamlit as st
import os
import io
import re
import tempfile
from translations import lang_db 

# مكتبات LangChain والذكاء الاصطناعي الأساسية
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# --- تحسينات الأداء: التخزين المؤقت للنماذج (Caching) ---

@st.cache_resource
def load_embeddings_model():
    """تحميل نموذج التضمين مرة واحدة فقط لتسريع الاستجابة"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm_model(api_key):
    """تثبيت نسخة الموديل في الذاكرة لمنع البطء"""
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=api_key)

#txt creation
def export_chat_to_txt(chat_history):
    chat_text = "Rai-Tech AI Chat History\n"
    chat_text += "="*30 + "\n\n"
    
    for msg in chat_history:
        role = "User" if msg.type == "human" else "Assistant"
        chat_text += f"{role}: {msg.content}\n"
        chat_text += "-"*20 + "\n"
        
    return chat_text.encode('utf-8')

# 1. إعداد واجهة الصفحة
st.set_page_config(page_title="Rai-Tech AI Assistant", page_icon="🤖")

# 2. تعريف اللغة أولاً
with st.sidebar:
    sidebar_title = lang_db["English"]["settings"] 
    st.header(sidebar_title)
    
    selected_lang = st.selectbox(
        "🌐 Language / اللغة / 言語", 
        ["English", "Arabic", "Japanese"],
        index=0 
    )

current_texts = lang_db[selected_lang]
api_key_secret = ""

try:
    if "GROQ_API_KEY" in st.secrets:
        api_key_secret = st.secrets["GROQ_API_KEY"]
except:
    api_key_secret = ""

with st.sidebar:
    if api_key_secret:
        api_key = api_key_secret
        st.success("✅ API Key loaded from Secrets")
    else:
        api_key = st.text_input(current_texts["api_key"], type="password")
    st.markdown("---")
    
    uploaded_pdf = st.file_uploader(current_texts["upload_pdf"], type="pdf")
    st.markdown("---")
        
    st.subheader(current_texts["action_title"])

    import urllib.parse
    company_phone = "81312345678" 
    whatsapp_msg = current_texts["whatsapp_msg"]
    encoded_msg = urllib.parse.quote(whatsapp_msg)
    
    whatsapp_url = f"https://wa.me/{81312345678}?text={encoded_msg}"

    st.markdown(f"""
        <a href="{whatsapp_url}" target="_blank" style="text-decoration: none;">
            <div style="
                background-color: #25D366;
                color: white;
                padding: 12px;
                text-align: center;
                border-radius: 8px;
                font-weight: bold;
                display: block;
                box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
            ">
                {current_texts["whatsapp_btn"]}
            </div>
        </a>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if "chat_history" in st.session_state and len(st.session_state.chat_history.messages) >=0:
        st.write(f"**{current_texts['export_title']}**")
        
        txt_data = export_chat_to_txt(st.session_state.chat_history.messages)
        
        st.download_button(
            label=current_texts["txt_download"], 
            data=txt_data, 
            file_name="rai_tech_chat.txt", 
            mime="text/plain", 
            use_container_width=True
        )
        st.markdown("---")
        if st.sidebar.button(current_texts["clear_chat"], use_container_width=True):
            if "chat_history" in st.session_state:
                st.session_state.chat_history.messages = []
            st.rerun()
            st.markdown("---")

    st.info(f"""
    **{current_texts['features_title']}**
    - {current_texts['f1']}
    - {current_texts['f2']}
    """)
    st.markdown("---")

    st.caption(current_texts["developer_info"])
    
st.title(current_texts["title"])
st.markdown("---")

if selected_lang == "Arabic":
    st.markdown("""
        <style>
        .main .block-container { direction: rtl; text-align: right; }
        .stChatInputContainer { direction: rtl; }
        .stChatMessage { text-align: right; direction: rtl; }
        .stChatMessage [data-testid="stChatMessageAvatar"] { margin-left: 10px; margin-right: 0px; }
        </style>
    """, unsafe_allow_html=True)

system_prompts = {
    "English": "You are a Rai-Tech Sales Assistant. ONLY use the provided filtered Context. Be concise and professional.",
    "Arabic": "أنت مساعد مبيعات في شركة Rai-Tech. استخدم فقط المعلومات المتوفرة في السياق. أجب باللغة العربية حصراً وبأسلوب مهذب.",
    "Japanese": "あなたはRai-Techの販売アシスタانتです。提供されたコンテキストのみを使用してください。丁寧な日本語で対応してください。"
}

if api_key:
    try:
        # --- تحسين: الـ Caching لعملية بناء الـ Vectorstore ---
        @st.cache_resource
        def init_rag(_pdf_file, _api_key):
            documents = TextLoader("company_data.txt", encoding="utf-8").load()
            if _pdf_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(_pdf_file.getbuffer())
                    tmp_path = tmp_file.name
                pdf_loader = PyPDFLoader(tmp_path)
                documents.extend(pdf_loader.load())
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
            final_docs = text_splitter.split_documents(documents)
            
            # استخدام الدالة المخبأة للـ Embeddings
            embeddings = load_embeddings_model()
            vectorstore = FAISS.from_documents(final_docs, embeddings)
            return vectorstore.as_retriever(search_kwargs={"k": 15})

        retriever = init_rag(uploaded_pdf, api_key)
        
        # استخدام الدالة المخبأة للـ LLM
        model = get_llm_model(api_key)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = ChatMessageHistory()

        if len(st.session_state.chat_history.messages) == 0:
            with st.chat_message("assistant"):
                st.markdown(current_texts["welcome_msg"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompts[selected_lang] + """
            Your ONLY source of information is the filtered Context provided.
            1. ONLY mention products that are present in the Context.
            2. If the Context contains a CRITICAL warning about no products, inform the user politely.
            3. Do not suggest products from your own memory.
            4. Be friendly but mathematically precise.
            5. If (Item_Price > USER_BUDGET) by even $1, Ignore it.
            6. Never suggest a product that costs more than the user's budget.
            7. If (Item_Price <= USER_BUDGET). Present it as a suitable options.
            8. When suggesting multiple products, always provide a summary table comparing their prices and key features.
            9. You may chat with customers and employees.
            10. If something is not clear, ask the user for clarification instead of guessing.
            11. Double-check the product name against its price before answering.
            Context: {context}"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])      

        def get_filtered_context(user_input, docs):
            budget_numbers = re.findall(r'(\d+)', user_input)
            if not budget_numbers:
                return "\n".join([doc.page_content for doc in docs])
            
            user_budget = float(budget_numbers[0])
            filtered_list = []
            found_any_product = False

            currency_pattern = r'(?:\$|USD|dollar|دولار|ドル)\s*(\d+)|(\d+)\s*(?:\$|USD|dollar|دولار|ドル)'

            for doc in docs:
                content = doc.page_content
                price_match = re.search(currency_pattern, content, re.IGNORECASE)
                
                if price_match:
                    price_str = price_match.group(1) if price_match.group(1) else price_match.group(2)
                    if price_str:
                        item_price = float(price_str)
                        if item_price <= user_budget:
                            filtered_list.append(f"PRODUCT: {content}")
                            found_any_product = True
                else:
                    filtered_list.append(f"INFO: {content}")

            if not found_any_product:
                no_match_msg = {
                    "English": "CRITICAL: No products found under this budget.",
                    "Arabic": "تنبيه: لا توجد منتجات تناسب هذه الميزانية في مخزوننا حالياً.",
                    "Japanese": "警告：この予算に合う商品はありません。"
                }
                return no_match_msg[selected_lang]
            
            return "\n".join(filtered_list)

        rag_chain = (
            {
                "context": lambda x: get_filtered_context(x["input"], retriever.invoke(x["input"])),
                "input": lambda x: x["input"],
                "history": lambda x: x["history"]
            }
            | prompt
            | model
        )

        full_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: st.session_state.chat_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        for msg in st.session_state.chat_history.messages:
            role = "user" if msg.type == "human" else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

        if user_query := st.chat_input(current_texts["input_placeholder"]):
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner(current_texts["spinner"]):
                    response = full_chain.invoke(
                        {"input": user_query},
                        config={"configurable": {"session_id": "any"}}
                    )
                    st.markdown(response.content)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning(current_texts["api_warning"])