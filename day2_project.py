from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from typing import List
import bs4
import os

os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(model="gpt-4o-mini")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def github_posts_loader(urls: list[str]) -> List[Document]:
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    return loader.load()


def split_docs(docs: list[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    return text_splitter.split_documents(docs)


def embedding_and_save_docs(docs: list[Document]) -> Chroma:
    return Chroma.from_documents(
        documents=docs, 
        embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )


def retrieve(vector_store: Chroma, query: str) -> List[Document]:
    return vector_store.as_retriever().invoke(query)


def relevancy_check(docs: List[Document], query: str) -> List[Document]:
    relevancy_check = False

    for doc in docs:
        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template="Answer yes or no to whether the query and the document are \"relevant\" to each other.\n{format_instructions}\n{query}\n",
            input_variables=["query", "document"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser
        anwser = chain.invoke({"query": query, "document": doc})

        relevancy_check |= anwser["relevant"] == "yes"

    return relevancy_check


def get_answer_from_docs(query: str, docs: List[Document]):
    prompt = hub.pull("rlm/rag-prompt")
    chain = prompt | llm | StrOutputParser()
    context = "\n\n".join(doc.page_content for doc in docs)

    return chain.invoke({"context": context, "question": query})


def check_hallucination(answer: str, query: str) -> bool:
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="Answer yes or no to whether answer has \"hallucinated\" the question.\n{format_instructions}\n",
        input_variables=["answer", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    anwser = chain.invoke({"answer": query, "question": query})

    return anwser["hallucinated"] == "yes"


if __name__ == "__main__":
    docs = github_posts_loader(urls)
    splitted_docs = split_docs(docs)
    vector_store = embedding_and_save_docs(splitted_docs)
    query = "What is the in-context learning in machine learning?"
    retrieved_docs = retrieve(vector_store, query)

    if not relevancy_check(retrieved_docs, query):
        print("No relevant documents found.")
        return

    for _ in range(2):
        answer = get_answer_from_docs(query, retrieved_docs)
        
        if not check_hallucination(answer, query):
            print(f"Query: {query}\n")
            print(f"Answer: {answer}\n")
            print("Referred Documents:")
            for doc in retrieved_docs:
                print(doc.metadata['source'])

    
