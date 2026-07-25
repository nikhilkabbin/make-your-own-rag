"""
LangChain-based RAG for ICC Cricket Dataset
Demonstrates semantic chunking, vector storage, and retrieval
"""
import os
from typing import List

import pandas as pd
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class ICCTournamentRAG:
    """LangChain RAG pipeline for tournament data"""

    def __init__(self, csv_path: str, cache_dir: str = "."):
        self.csv_path = csv_path
        self.cache_dir = cache_dir
        self.vectorstore = None
        self.retriever = None
        self.llm = None

        # Model configuration
        self.embedding_model = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
        self.language_model = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

    def load_and_chunk_csv(self) -> List:
        """
        Load CSV and create semantic chunks instead of raw rows
        This is the key difference from naive RAG
        """
        df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
        documents = []

        for idx, row in df.iterrows():
            if not row.get('Year') or not row.get('Winner'):
                continue

            year = str(row['Year']).strip()
            tournament = str(row['Tournament']).strip()
            winner = str(row['Winner']).strip()
            runner_up = str(row['Runner-up']).strip()
            venue = str(row['Venue']).strip()

            # Create three semantic variations of the same data
            # This increases retrieval robustness
            chunks = [
                # Variation 1: Winner-focused (best for "who won" queries)
                f"{tournament} in {year} was won by {winner}. Runner-up was {runner_up}.",

                # Variation 2: Year/venue-focused (best for "what happened in YEAR" queries)
                f"In {year}, {tournament} was held in {venue}. {winner} defeated {runner_up} in the final.",

                # Variation 3: Comprehensive (fallback for complex queries)
                f"{tournament} {year} ({venue}): Winner {winner}, Runner-up {runner_up}.",
            ]

            for chunk_text in chunks:
                # LangChain Document object wraps text + metadata
                doc = {
                    'page_content': chunk_text,
                    'metadata': {
                        'year': year,
                        'tournament': tournament,
                        'winner': winner,
                        'source': f"row_{idx}",
                    }
                }
                documents.append(doc)

        print(f"✓ Created {len(documents)} semantic chunks from {len(df)} rows")
        return documents, df

    def setup_vectorstore(self, force_rebuild: bool = False):
        """
        Initialize embeddings and vector store (FAISS)
        LangChain handles all the plumbing
        """
        cache_path = os.path.join(self.cache_dir, "vectorstore")

        # Try to load cached vectorstore
        if os.path.exists(cache_path) and not force_rebuild:
            print("✓ Loading cached vectorstore...")
            self.vectorstore = FAISS.load_local(
                cache_path,
                OllamaEmbeddings(model=self.embedding_model),
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            return

        # Otherwise, create new vectorstore
        print("Creating vectorstore (this embeds all chunks)...")

        # Load and chunk CSV
        docs, _ = self.load_and_chunk_csv()

        # Initialize embeddings
        embeddings = OllamaEmbeddings(model=self.embedding_model)

        # Create FAISS vectorstore from documents
        # LangChain handles batch embedding automatically
        texts = [doc['page_content'] for doc in docs]
        metadatas = [doc['metadata'] for doc in docs]

        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )

        # Save to disk for future loads
        self.vectorstore.save_local(cache_path)
        print(f"✓ Vectorstore saved to {cache_path}")

        # Create retriever with k=5 top results
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def setup_chain(self):
        """
        Build the RAG chain: retriever → formatter → LLM
        LangChain's pipe syntax makes this clean
        """
        self.llm = ChatOllama(model=self.language_model)

        # Define the prompt template
        # This tells the LLM how to use retrieved context
        prompt_template = """You are a cricket expert analyzing ICC tournament data.

        Use only the following tournament information to answer the question:

        {context}

        User question: {question}

        If the data doesn't contain the answer, say: "This information is not in my dataset."
        Be concise and direct."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Build the chain using LangChain's pipe syntax
        # retriever.map() formats retrieved docs as strings
        # prompt receives context + question
        # llm generates response
        # output_parser extracts text

        def format_docs(docs):
            """Format retrieved documents as context"""
            return "\n".join([
                f"- (match: {doc.metadata.get('tournament', 'N/A')} {doc.metadata.get('year', 'N/A')}) {doc.page_content}"
                for doc in docs
            ])

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> str:
        """
        Run a query through the RAG pipeline
        """
        if not self.chain:
            raise ValueError("Chain not initialized. Call setup_chain() first.")

        # Invoke the chain
        response = self.chain.invoke(question)

        # Also get the retrieved documents for debugging
        docs = self.retriever.invoke(question)

        return response, docs

    def interactive_mode(self):
        """
        Run an interactive chat loop
        """
        print("\n" + "="*60)
        print("ICC Cricket Tournament RAG")
        print("Try: 'Who won the 2024 T20 World Cup?'")
        print("     'What happened in 2023?'")
        print("     'Who won in 2025?'")
        print("Type 'exit' to quit\n" + "="*60 + "\n")

        while True:
            question = input("Q: ").strip()
            if question.lower() == 'exit':
                break

            if not question:
                continue

            # Get response and retrieved context
            response, retrieved_docs = self.query(question)

            # Show retrieved context for transparency
            print("\nContext retrieved:")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"  [{i}] {doc.page_content}")

            # Show LLM response
            print(f"\nA: {response}\n")


def main():
    """Initialize and run the RAG system"""

    # Path to your ICC dataset
    csv_path = 'icc_dataset.csv'

    # Initialize RAG
    rag = ICCTournamentRAG(csv_path, cache_dir='.')

    print("Setting up vectorstore...")
    rag.setup_vectorstore()

    print("Setting up LLM chain...")
    rag.setup_chain()

    # Run interactive mode
    rag.interactive_mode()


if __name__ == "__main__":
    main()
