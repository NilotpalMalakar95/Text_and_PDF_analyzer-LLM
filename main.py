import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA

file_path = "0. Data/medium_blog_2.txt"

if __name__ == "__main__":
    # print('Hello Vector DB')

    # Instantiating the document_loader
    loader = TextLoader(file_path, encoding="utf-8")
    # Loading the document data
    document = loader.load()

    # Splitting the text data
    # Instantiating the splitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # Chunk size defines the size of each split the splitter would generate
    # Chunk Overlap would define the amount of shared tokens amongst consecutive chunks so that the context can be preserved in each split even after breaking the document into pieces
    text_splits = text_splitter.split_documents(document)

    # Embedding the chunks
    # Initializing the embeddings object
    embedding_object = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # Embed and store the text splits to pinecone -> Creating a document search object
    document_search_object = Pinecone.from_documents(
        text_splits, embedding_object, index_name="medium-blogs-embeddings-index"
    )
    # This document search object would take the text chunks, perform embedding using the embedding object and store these embeddings/vectors into the vector db
    # The index name is the name of the index that we set in pinecone.io while creating the index; It will also need a dimension tat can be found from the dimension of output embeddings of the embeddings object, here :-> for OpenAIEmbeddings the output dim is 1536; source : openai documentation

    # Instantiating the qa object
    # Takes in a model, type of chain to be created and the retriever object
    # chain_type = 'stuff' means, the relevant chunks been identified would be taken as it is and plugged into the query as it is
    # This qa object would do the following :
    # 1. Takes in the query
    # 2. Vectorizes it
    # 3. Searches for the best possible matching n documents vector form the vector db
    # 4. Fetches these n document vectors
    # 5. Converts these vectors back to documents
    # 6. plugs in these fetched documents into the original query as context as it is
    qa_object = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=document_search_object.as_retriever(),
        return_source_documents=True,
    )

    # Query to resolve from the document
    query = "What is a vector database ? Give me a 15 words answer understandable for a begginner."

    # Querying the qa_object
    result = qa_object.invoke({"query": query})

    print(result)
