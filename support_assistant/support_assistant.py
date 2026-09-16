import os
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from fastapi import FastAPI


# Task 1
docs_folder = "docs"
db_folder = "chroma_db"

documents = {
    "doc_01.txt": (
        "Zepto delivers grocery and household essentials to serviceable pin codes "
        "within 10 to 30 minutes of order confirmation, depending on the customer's "
        "delivery zone and current order volume. Standard delivery is free on orders over "
        "INR 149; orders below this threshold incur a flat INR 25 delivery fee. Priority "
        "delivery, which reserves the next available rider slot, is available at checkout "
        "for an additional INR 15. Zepto does not currently deliver to addresses outside "
        "its listed serviceable pin codes."
    ),

    "doc_02.txt": (
        "Grocery and perishable items may be reported for a return within 24 hours of delivery "
        "if damaged, spoiled, or incorrect; non-perishable packaged items may be returned "
        "within 7 days of delivery in unopened, resalable condition. Approved refunds are "
        "credited to the original payment method within 3–5 business days, or instantly to "
        "the Zepto wallet if the customer opts for wallet credit. Personal care items that "
        "have been opened are non-returnable except in the case of a manufacturing defect. "
        "Return pickup, where required, is arranged free of cost by Zepto."
    ),

    "doc_03.txt": (
        "Zepto offers three account tiers: Basic (free, default tier, standard delivery fees apply), "
        "Zepto Pass (INR 49 per month, free standard delivery on all orders and 5% off select categories), "
        "and Zepto Pass+ (INR 99 per month, free priority delivery, 10% off select categories, and early "
        "access to limited-time deals 24 hours before they go live to Basic and Pass members). "
        "Membership can be cancelled at any time from account settings; cancelling stops the next "
        "billing cycle but does not refund the current membership period."
    ),

    "doc_04.txt": (
        "Every Zepto order shows a live rider-tracking map from the moment it is packed until delivery, "
        "accessible from the 'Track Order' screen. Estimated delivery time updates automatically as the "
        "rider moves. If an order's status shows no movement for more than 20 minutes past its original "
        "estimated delivery time, customers should contact support directly rather than continue waiting, "
        "since this indicates a likely delivery issue."
    ),

    "doc_05.txt": (
        "Orders can be cancelled free of cost any time before the order status changes to 'Packed', "
        "typically within the first 2 minutes of placing the order. Once an order has been packed, "
        "it can no longer be cancelled through the app, since the rider is dispatched immediately "
        "after packing given Zepto's quick-delivery model. If a packed order cannot be delivered due "
        "to a Zepto-side issue (for example, rider unavailability), the order is auto-cancelled and "
        "fully refunded without any cancellation fee."
    ),

    "doc_06.txt": (
        "If an order arrives with damaged, spoiled, or missing items, customers must report it within "
        "24 hours of delivery through the 'Report an Issue' button on the order page. Zepto ships a "
        "free replacement or issues a full refund for damaged, spoiled, or missing items without "
        "requiring the customer to return the original item, unless the order value exceeds INR 1000, "
        "in which case a photo of the issue must be submitted through the report form before a "
        "replacement or refund is processed."
    ),

    "doc_07.txt": (
        "Zepto gift cards are available in fixed denominations of INR 100, INR 250, INR 500, and "
        "INR 1000, and are delivered by email or SMS within minutes of purchase. Gift cards are valid "
        "for 1 year from the date of issue and carry no maintenance fees. Gift card balance can be "
        "combined with one other payment method at checkout but cannot be combined with another gift "
        "card in the same transaction. Gift card balance cannot be redeemed for cash except where "
        "required by law."
    ),

    "doc_08.txt": (
        "Zepto customer support is available via in-app chat 24 hours a day, 7 days a week, given "
        "the time-sensitive nature of quick commerce deliveries. Average in-app chat response time "
        "is under 2 minutes. Email support is also available for non-urgent queries and is answered "
        "within 24 hours on business days. Phone support is not offered."
    )
}


os.makedirs(docs_folder, exist_ok=True)

for file_name, text in documents.items():

    file_path = os.path.join(docs_folder, file_name)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

print("Policy documents created.")


model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
ids = []
metadatas = []

for file_name in sorted(os.listdir(docs_folder)):
    if file_name.endswith(".txt"):
        file_path = os.path.join(docs_folder, file_name)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

        words = text.split()
        chunk_size = 40

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])

            texts.append(chunk)
            ids.append(f"{file_name[:-4]}_chunk_{i // chunk_size + 1}")
            metadatas.append({"source": file_name})

print("Documents loaded:", len(set(metadatas[i]["source"] for i in range(len(metadatas)))))
print("Total chunks:", len(texts))

embeddings = model.encode(texts).tolist()

print("Embeddings created.")
print("Embeddings created.")


os.makedirs(db_folder, exist_ok=True)

client = chromadb.PersistentClient(path=db_folder)

try:
    client.delete_collection("zepto_policies")
except Exception:
    pass


collection = client.create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)


collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Documents stored in ChromaDB:", collection.count())

#Task 2
prompt= """
You are a Zepto customer support assistant.

Use the information given in the context to answer the customer's question.

Context:
{context}

Customer question:
{question}

Give a clear and short answer based on the context.
Keep the answer within 2 to 4 sentences.

Do not use information that is not present in the given context.
If the context does not contain the answer, say that the information is not available.

Example:

Context:
Zepto gift cards are valid for 1 year from the date of issue.

Question:
How long is a Zepto gift card valid?

Answer:
A Zepto gift card is valid for 1 year from the date of issue.

Now answer the customer's question using the context above.
"""

#Task3
MOCK_LLM = os.getenv("MOCK_LLM", "1")


class SupportState(TypedDict):
    question: str
    intent: str
    answer: dict


def classify_intent(state: SupportState):

    question = state["question"].lower()

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    if any(word in question for word in keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {"intent": intent}

#Task4
class SupportResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

def generate_real_llm_answer(prompt_text, retries=2):

    for attempt in range(retries + 1):

        try:
            answer = "Real LLM answer goes here."

            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Invalid LLM answer")

            return answer

        except Exception as error:

            if attempt == retries:
                raise error

    raise ValueError("Unable to generate a valid answer")


def retrieve_and_answer(state: SupportState):

    question = state["question"]

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_docs = results["documents"][0]
    retrieved_ids = results["ids"][0]

    if MOCK_LLM != "0":

        top_chunk = retrieved_docs[0]

        response = SupportResponse(
            answer=f"Based on the retrieved context: {top_chunk[:200]}",
            sources=retrieved_ids,
            confidence=1.0
        )

        return {"answer": response.model_dump()}

    context = "\n\n".join(retrieved_docs)

    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    answer = generate_real_llm_answer(formatted_prompt)


    response = SupportResponse(
        answer=answer,
        sources=retrieved_ids,
        confidence=1.0
    )

    return {"answer": response.model_dump()}


def direct_answer(state: SupportState):

    if MOCK_LLM != "0":

        response = SupportResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )

        return {"answer": response.model_dump()}

    formatted_prompt = prompt.format(
        context="",
        question=state["question"]
    )

    answer = generate_real_llm_answer(formatted_prompt)


    response = SupportResponse(
        answer=answer,
        sources=[],
        confidence=1.0
    )

    return {"answer": response.model_dump()}


def route_question(state: SupportState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"

graph = StateGraph(SupportState)

graph.add_node("classify_intent", classify_intent)
graph.add_node("retrieve_and_answer", retrieve_and_answer)
graph.add_node("direct_answer", direct_answer)

graph.add_edge(START, "classify_intent")

graph.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph.add_edge("retrieve_and_answer", END)
graph.add_edge("direct_answer", END)

app = graph.compile()

#Task5
class AskRequest(BaseModel):
    query: str


api = FastAPI()


@api.post("/ask", response_model=SupportResponse)
def ask_question(request: AskRequest):

    result = app.invoke({
        "question": request.query,
        "intent": "",
        "answer": {}
    })

    return result["answer"]

# Test examples for LangGraph routing
policy_test = app.invoke({
    "question": "How can I track my delivery?",
    "intent": "",
    "answer": {}
})

general_test = app.invoke({
    "question": "What is the capital of India?",
    "intent": "",
    "answer": {}
})

print("\nPolicy Question Test:")
print(policy_test)

print("\nGeneral Question Test:")
print(general_test)