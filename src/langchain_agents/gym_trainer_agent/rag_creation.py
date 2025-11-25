import os

from dotenv import load_dotenv
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVector, PGVectorStore

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_paths = [
    os.path.join(current_dir, "data", "refactored_gym_training.md"),
    os.path.join(current_dir, "data", "refactored_informe_rutina.md"),
]
persistent_directory = os.path.join(current_dir, "vectorstore", "chroma_gym_db")


def load_and_chunk_documents(file_paths):
    """
    Load markdown documents and split them into chunks using header-based and character-based splitting
    """
    # Validate files exist
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    # Load documents
    loader = DirectoryLoader(
        "./data/",
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
    )
    docs = loader.load()

    # Initialize header-based splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )

    # Split each document individually
    split_docs = []
    for doc in docs:
        markdown_text = doc.page_content  # Extract markdown string
        chunks = markdown_splitter.split_text(
            markdown_text
        )  # Returns list of Documents
        # Optional: preserve original metadata
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
        split_docs.extend(chunks)

    # Initialize character-based splitter for large chunks
    secondary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
    )

    # Further split large chunks
    fine_chunks = []
    for doc in split_docs:
        if len(doc.page_content) > 500:
            fine_chunks.extend(secondary_splitter.split_documents([doc]))
        else:
            fine_chunks.append(doc)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{fine_chunks[0].page_content}\n")

    return fine_chunks


# CHORMA
if not os.path.exists(persistent_directory):
    print("Persistance directory does not exist. Initializing vector store...")

    fine_chunks = load_and_chunk_documents(file_paths)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        fine_chunks,
        embeddings,
        persist_directory=persistent_directory,
    )

else:
    print("Vector store already exists. No need to initialize.")

# PGVECTOR
try:
    print("Initializing PGVector...")
    CONNECTION_STRING = "postgresql+psycopg://postgres:test@localhost:5432/gym_vdb"
    # engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

    # VECTOR_SIZE = 768
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    TABLE_NAME = "gym_docs"
    # engine.init_vectorstore_table(
    #     table_name=TABLE_NAME,
    #     vector_size=VECTOR_SIZE,
    # )

    pgvectorstore = PGVector(
        embeddings=embedding,
        collection_name=TABLE_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )
    print("PGVector initialized successfully.")

    fine_chunks = load_and_chunk_documents(file_paths)
    pgvectorstore.add_documents(fine_chunks)

except Exception as e:
    print(f"Error in pgvector creation: {e}")
