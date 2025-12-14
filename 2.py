from typing import List
from pydantic import BaseModel
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing_extensions import Literal
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage,SystemMessage

#------

st.title("🦜 Route-chaining RAG LangGraph")
st.subheader("This is a Cash Memo, Invoice, Quotation RAG Application")
st.sidebar.title("Azure OpenAI Settings")

AZURE_OPENAI_API_KEY = st.sidebar.text_input("API Key", type="password")
AZURE_ENDPOINT_URI = st.sidebar.text_input("Azure Endpoint")
API_VERSION = st.sidebar.text_input("API Version")
DEPLOYMENT = st.sidebar.text_input("Deployment Name")

if not (AZURE_OPENAI_API_KEY and AZURE_ENDPOINT_URI and API_VERSION and DEPLOYMENT):
    st.warning("⚠ Please enter all Azure resource info.")
    st.stop()

llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_ENDPOINT_URI,
    api_version=API_VERSION,
    azure_deployment=DEPLOYMENT,
    temperature=0.0)


file = st.file_uploader("Upload a PDF file to train & analyze:")
user_question = st.chat_input("Ask a question about the file:")

# ----

if st.button("Learn Document") and file:
    with st.spinner("Reading and embedding document..."):
        file_path = f"./uploaded.pdf"
        with open(file_path, "wb") as f:
            f.write(file.read())
        docs = PyPDFLoader(file_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.retriever = vectorstore.as_retriever()
    st.success("📚 Document learned successfully! Now you can ask questions.")

class Route(BaseModel):
    step:Literal["Cash_Memo","Invoice","Quotation"]=Field(description="The next step in the routing process") # llm can return either cashmemo, invoice or quotation
router=llm.with_structured_output(Route)

class RAGState(TypedDict):
    question: str = ""
    decision: str = ""
    answer: str = ""
    #retrieved_docs: List[Document] = []
    

def llm_call_router(state: RAGState):
    """Route the input to the appropriate node"""
    retriever = st.session_state.retriever
    docs = retriever.invoke("Figure out each and everything from the document. Is it a cash memo or invoice or Quotation? Figure out everything") #(state["question"])
    context = "\n\n".join([d.page_content for d in docs])
    decision=router.invoke([SystemMessage(content="Route the input to Cash_Memo, Invoice, Quotation based on the provided documents"),
                            HumanMessage(content=f"QUESTION: {state['question']}\n\nCONTEXT:\n{context}")]) # state["retrieved_docs"])])
    return {"decision":decision.step}


def file_type(state: RAGState):
    if state["decision"] == "Cash_Memo":
        return "Cash_Memo"
    elif state["decision"] == "Invoice":
        return "Invoice"
    elif state["decision"] == "Quotation":
        return "Quotation"

# ----------

def Cash_Memo(state: RAGState):
    """Fetch information from a Cash Memo"""
    retriever = st.session_state.retriever
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
    You are a Cash Memo document reader. Answer using ONLY the context below.
    If answer is not present, say "Not found in document".
    About Cash Memo:
    It is Issued when a customer makes an immediate cash purchase. It acts as proof of payment and delivery of goods/services.
    It displays items purchased, quantity, price, and total amount. Indicates that payment has already been made (usually in cash).
    No mention of credit terms or due dates. It is used after Retail transactions or instant payments.
    These are all informations about cash memo. Now accordingly fetch information from the CONTEXT provided as per QUESTION provided.

    CONTEXT:
    {context}

    QUESTION:
    {state["question"]}
    """
    response = llm.invoke(prompt).content
    return {"answer": response}
    #return RAGState(question=state.question, file_type=state.file_type, retrieved_docs=docs, answer=response)

# ----------

def Invoice(state: RAGState):
    """Fetch information from a Invoice"""
    retriever = st.session_state.retriever
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
    You are a Invoice document reader. Answer using ONLY the context below.
    If answer is not present, say "Not found in document".
    About Invoice:
    It is formal request for payment after goods/services are delivered. It is legally binding document for accounts and taxation.
    It displays invoice number, date, buyer/seller details, itemized list, taxes, and total amount. 
    Specifies payment terms (due date, mode of payment). It is used for both cash and credit transactions.
    It is used after sale is confirmed and goods/services are delivered.
    These are all informations about Invoice. Now accordingly fetch information from the CONTEXT provided as per QUESTION provided.

    CONTEXT:
    {context}

    QUESTION:
    {state["question"]}
    """
    response = llm.invoke(prompt).content
    return {"answer": response}
    #return RAGState(question=state.question, file_type=state.file_type, retrieved_docs=docs, answer=response)

# ----------

def Quotation(state: RAGState):
    """Fetch information from a Quotation"""
    retriever = st.session_state.retriever
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
    You are a Quotation document reader. Answer using ONLY the context below.
    If answer is not present, say "Not found in document".
    About Quotation:
    It is a price estimate provided before the sale. It is a Non-binding document offering details of products/services and 
    their prices. It includes item descriptions, unit prices, and total estimated cost. Often mentions validity period 
    (e.g., “Valid for 30 days”). It does not confirm a sale; it's an offer for negotiation. It is used before the customer 
    agrees to purchase. These are all informations about Quotation. Now accordingly fetch information from the CONTEXT provided 
    as per QUESTION provided.

    CONTEXT:
    {context}

    QUESTION:
    {state["question"]}
    """
    response = llm.invoke(prompt).content
    return {"answer": response}
    #return RAGState(question=state.question, file_type=state.file_type, retrieved_docs=docs, answer=response)


# ----------

router_builder = StateGraph(RAGState)
router_builder.add_node("llm_call_router", llm_call_router)
router_builder.add_node("Cash_Memo", Cash_Memo)
router_builder.add_node("Invoice", Invoice)
router_builder.add_node("Quotation", Quotation)
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    file_type,
    {"Cash_Memo": "Cash_Memo",
    "Invoice": "Invoice",
    "Quotation": "Quotation"})
# Name returned by route_decision : Name of next node to visit
router_builder.add_edge("Cash_Memo", END)
router_builder.add_edge("Invoice", END)
router_builder.add_edge("Quotation", END)
router_workflow = router_builder.compile()

# ----------
png_bytes = router_workflow.get_graph().draw_mermaid_png()
st.image(png_bytes)

# ----------

if user_question:
    if "retriever" not in st.session_state:
        st.warning("Upload the file, select the file type and click Learn Document first.")
    else:
        final_state = router_workflow.invoke({"question":user_question})
        st.write("Your Question: ", user_question)
        st.success(final_state["answer"])
