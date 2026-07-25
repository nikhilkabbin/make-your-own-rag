import os
import csv

import ollama
import pickle


EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))


def load_csv_smart(filepath):
    """
    Load CSV with BOM handling and create semantic chunks.
    """
    chunks = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with missing critical data
            if not row.get('Year') or not row.get('Winner'):
                continue

            year = row['Year'].strip()
            tournament = row['Tournament'].strip()
            winner = row['Winner'].strip()
            runner_up = row['Runner-up'].strip()
            venue = row['Venue'].strip()

            # Create multiple semantic chunks from one row

            # Chunk 1: Winner-focused (best for "who won" queries)
            chunk1 = f"{tournament} in {year} was won by {winner}. Runner-up was {runner_up}."
            chunks.append(chunk1)

            # Chunk 2: Year-focused (best for "what happened in YEAR" queries)
            chunk2 = f"In {year}, {tournament} was held in {venue}. {winner} defeated {runner_up} in the final."
            chunks.append(chunk2)

            # Chunk 3: Venue-focused (best for "tournaments in PLACE" queries)
            chunk3 = f"{tournament} {year} was held in {venue}. Winner: {winner}. Runner-up: {runner_up}."
            chunks.append(chunk3)

    return chunks


def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0


def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def save_vector_db(filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(VECTOR_DB, f)


def load_vector_db(filepath):
    global VECTOR_DB
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            VECTOR_DB = pickle.load(f)
        return True
    return False

def main():
    cache_file = 'icc_vector_db.pkl'
    if not load_vector_db(cache_file):
        print("Loading CSV and indexing...")
        dataset = load_csv_smart('icc_dataset.csv')
        print(f"Created {len(dataset)} semantic chunks")

        for i, chunk in enumerate(dataset):
            add_chunk_to_database(chunk)
            if (i + 1) % 20 == 0:
                print(f"  Indexed {i+1}/{len(dataset)}")

        save_vector_db(cache_file)
        print("✓ Cached to disk\n")
    else:
        print(f"✓ Loaded {len(VECTOR_DB)} cached embeddings\n")

    print("=" * 60)
    print("ICC World Cup RAG System")
    print("Try: 'Who won the 2024 T20 World Cup?'")
    print("     'What happened in 2023?'")
    print("     'Who won in 2025?'")
    print("Type 'exit' to quit\n" + "=" * 60 + "\n")


    while True:
        query = input('Q: ')
        if query.lower() == 'exit':
            break

        retrieved_knowledge = retrieve(query, top_n=5)

        print('\nContext found:')
        for chunk, similarity in retrieved_knowledge:
            print(f'  [{similarity:.3f}] {chunk}')

        context = '\n'.join([f'- {chunk}' for chunk, _ in retrieved_knowledge])

        instruction_prompt = f"""You are a cricket expert assistant analyzing ICC tournament data.

        Answer the user's question using ONLY the following tournament information:

        {context}

        If the data doesn't contain the answer, say: "This information is not in my dataset."
        Be concise and direct."""

        stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': query},
            ],
            stream=True,
        )

        print('\nA: ', end='')
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print("\n")


if __name__ == "__main__":
    main()
