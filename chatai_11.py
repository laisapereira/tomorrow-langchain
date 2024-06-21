from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# chunks semelhantes à query do usuário.
def similar_chunks(user_query):
    client = chromadb.PersistentClient(path="./tomorrow")
    col = client.get_or_create_collection(
        "langchain",
        embedding_function=OpenAIEmbeddingFunction(api_key=os.getenv('OPENAI_API_KEY'))
    )
    results = col.query(query_texts=[user_query], n_results=100)
    return results['documents'][0], results['embeddings'][0]

# similaridade coseno para reranquear os chunks retornados pela busca padrão (acima)

""" def rerank_chunks(query_embedding, chunks_embeddings, chunks):
    similarities = cosine_similarity([query_embedding], chunks_embeddings)[0]
    ranked_chunks = [chunk for _, chunk in sorted(zip(similarities, chunks), key=lambda x: x[0], reverse=True)]
    return ranked_chunks[:10] """

def prompt_llm(user_query, chunks):
    model = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Quero que você seja um expert em quimica organica, inteligencia artificial e design de código."),
        ("user", "{user_query}"),
        ("system", "As descrições sobre os temas necessários estão em: {cdd_info}. Se limite a responder com base nessas informações fornecidas. Tente não trazer outras informações na sua resposta."),
        ("system", "Responda em até 150 palavras."),
    ])

    chain = prompt | model 

    response = chain.invoke({
        "user_query": user_query,
        "cdd_info": chunks
    })
    return response.content

history = []

def chat_user(user_query):
    history.append(user_query)

    chunks, chunks_embeddings = similar_chunks(user_query)
    query_embedding = OpenAIEmbeddingFunction(api_key=os.getenv('OPENAI_API_KEY')).embed(user_query)
    ranked_chunks = rerank_chunks(query_embedding, chunks_embeddings, chunks)

    response = prompt_llm(user_query, "\n".join(ranked_chunks))

    history.append(response)
    print(response)
    print(history)
    return response 
