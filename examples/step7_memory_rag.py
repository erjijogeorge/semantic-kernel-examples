"""
Step 7: Memory & RAG (Retrieval Augmented Generation)

This example demonstrates how to give AI long-term memory and the ability to
search and retrieve information from documents.

Concepts:
- Vector databases for semantic search
- Storing and retrieving memories
- Document chunking and ingestion
- RAG pattern: Retrieve relevant context → Augment prompt → Generate answer
- Semantic search (find similar content)
- Building Q&A systems over your documents

Key Components:
- ChromaDB: Vector database for storing embeddings
- Embeddings: Convert text to numerical vectors
- Semantic Search: Find similar content by meaning, not keywords
- RAG: Enhance AI responses with retrieved context
"""

import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Vector database
import chromadb
from chromadb.config import Settings

load_dotenv()


async def example1_basic_memory_storage():
    """Example 1: Basic memory storage and retrieval"""
    print("💾 Example 1: Basic Memory Storage & Retrieval")
    print("-" * 70)
    
    # Initialize ChromaDB (in-memory for this example)
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))
    
    # Create a collection (like a table in a database)
    collection = client.create_collection(
        name="memories",
        metadata={"description": "User memories and facts"}
    )
    
    # Store some memories (facts about a user)
    memories = [
        "My name is Alice and I'm a software engineer.",
        "I love Python programming and AI.",
        "I have a dog named Max who is 3 years old.",
        "My favorite food is sushi.",
        "I live in San Francisco and work remotely.",
        "I enjoy hiking on weekends.",
        "I'm learning to play the guitar.",
    ]
    
    print("Storing memories in vector database...")
    
    # Add memories to the collection
    # ChromaDB will automatically create embeddings
    collection.add(
        documents=memories,
        ids=[f"mem_{i}" for i in range(len(memories))],
        metadatas=[{"type": "user_fact"} for _ in memories]
    )
    
    print(f"✅ Stored {len(memories)} memories\n")
    
    # Query the memories
    queries = [
        "What is the user's profession?",
        "Does the user have any pets?",
        "What are the user's hobbies?",
    ]
    
    for query in queries:
        print(f"🔍 Query: {query}")
        
        # Semantic search - find most relevant memories
        results = collection.query(
            query_texts=[query],
            n_results=2  # Get top 2 most relevant
        )
        
        print(f"📝 Relevant memories:")
        for i, doc in enumerate(results['documents'][0], 1):
            print(f"   {i}. {doc}")
        print()
    
    # Cleanup
    client.delete_collection("memories")


async def example2_document_qa_with_rag():
    """Example 2: Document Q&A using RAG pattern"""
    print("📚 Example 2: Document Q&A with RAG")
    print("-" * 70)
    
    # Initialize kernel and services
    kernel = Kernel()
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    service_id = "chat"
    chat_service = AzureChatCompletion(
        service_id=service_id,
        api_key=api_key,
        endpoint=endpoint,
        deployment_name=deployment_name,
    )
    kernel.add_service(chat_service)
    
    # Initialize ChromaDB
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))
    
    collection = client.create_collection(name="documents")
    
    # Sample documents about AI and Machine Learning
    documents = [
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines. It includes learning, reasoning, and self-correction.",
        "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
        "Deep Learning is a subset of machine learning based on artificial neural networks with multiple layers.",
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
        "Computer Vision is a field of AI that trains computers to interpret and understand the visual world using digital images and videos.",
        "Reinforcement Learning is a type of machine learning where an agent learns to make decisions by performing actions and receiving rewards.",
        "Supervised Learning uses labeled data to train models, while Unsupervised Learning finds patterns in unlabeled data.",
        "Neural Networks are computing systems inspired by biological neural networks that constitute animal brains.",
    ]
    
    print("📥 Ingesting documents into vector database...")
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"source": "AI_knowledge_base"} for _ in documents]
    )
    print(f"✅ Ingested {len(documents)} documents\n")

    # RAG Pattern: Answer questions using retrieved context
    questions = [
        "What is the difference between supervised and unsupervised learning?",
        "What is Deep Learning?",
        "How does Natural Language Processing work?",
    ]

    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.3,  # Lower temperature for factual answers
        max_tokens=300,
    )

    for question in questions:
        print(f"❓ Question: {question}")

        # Step 1: RETRIEVE - Find relevant documents
        results = collection.query(
            query_texts=[question],
            n_results=3  # Get top 3 most relevant documents
        )

        relevant_docs = results['documents'][0]
        context = "\n".join(relevant_docs)

        print(f"📄 Retrieved context ({len(relevant_docs)} documents)")

        # Step 2: AUGMENT - Create prompt with retrieved context
        prompt = f"""Answer the following question using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        # Step 3: GENERATE - Get AI response
        result = await kernel.invoke_prompt(
            prompt=prompt,
            settings=execution_settings
        )

        print(f"🤖 Answer: {result}\n")

    # Cleanup
    client.delete_collection("documents")


async def example3_conversational_memory():
    """Example 3: Conversational memory - AI remembers past conversations"""
    print("💬 Example 3: Conversational Memory")
    print("-" * 70)

    kernel = Kernel()

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    service_id = "chat"
    chat_service = AzureChatCompletion(
        service_id=service_id,
        api_key=api_key,
        endpoint=endpoint,
        deployment_name=deployment_name,
    )
    kernel.add_service(chat_service)

    # Initialize memory store
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))

    memory_collection = client.create_collection(name="conversation_memory")

    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.7,
        max_tokens=300,
    )

    # Simulate a conversation where we store facts
    print("Simulating a conversation where AI learns about the user...\n")

    conversation_turns = [
        "Hi! My name is Bob and I'm a data scientist.",
        "I'm working on a project about climate change prediction.",
        "I use Python and TensorFlow for my work.",
        "What do you remember about me?",
    ]

    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a helpful assistant with memory. Remember important facts about the user."
    )

    memory_id = 0

    for user_message in conversation_turns:
        print(f"👤 User: {user_message}")
        chat_history.add_user_message(user_message)

        # If this is a question about memory, retrieve relevant memories
        if "remember" in user_message.lower() or "know about" in user_message.lower():
            # Retrieve memories
            results = memory_collection.query(
                query_texts=[user_message],
                n_results=5
            )

            if results['documents'][0]:
                memories = "\n".join(results['documents'][0])
                context_message = f"Here's what I remember about you:\n{memories}"
                chat_history.add_system_message(context_message)
                print(f"🧠 [Retrieved {len(results['documents'][0])} memories]")

        # Get AI response
        response = await chat_service.get_chat_message_content(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel
        )

        assistant_message = str(response)
        chat_history.add_assistant_message(assistant_message)
        print(f"🤖 Assistant: {assistant_message}")

        # Store important facts from user messages in memory
        if memory_id < 3:  # Store first 3 messages as memories
            memory_collection.add(
                documents=[user_message],
                ids=[f"memory_{memory_id}"],
                metadatas=[{"type": "user_fact"}]
            )
            memory_id += 1

        print()

    # Cleanup
    client.delete_collection("conversation_memory")


async def example4_semantic_search_demo():
    """Example 4: Semantic search - find by meaning, not keywords"""
    print("🔍 Example 4: Semantic Search Demo")
    print("-" * 70)

    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))

    collection = client.create_collection(name="semantic_search")

    # Add diverse documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a popular programming language for data science.",
        "Machine learning models can predict future trends.",
        "The weather today is sunny and warm.",
        "Neural networks are inspired by the human brain.",
        "I love eating pizza on Friday nights.",
        "Quantum computing could revolutionize cryptography.",
        "The cat sat on the mat and purred contentedly.",
    ]

    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))],
    )

    print("📚 Document collection created\n")

    # Semantic search examples
    searches = [
        ("artificial intelligence", "Search for: 'artificial intelligence' (not in any document!)"),
        ("coding", "Search for: 'coding' (synonym for programming)"),
        ("pets", "Search for: 'pets' (related to animals)"),
    ]

    for query, description in searches:
        print(f"🔎 {description}")
        results = collection.query(
            query_texts=[query],
            n_results=2
        )

        print(f"   Top results:")
        for i, doc in enumerate(results['documents'][0], 1):
            print(f"   {i}. {doc}")
        print()

    print("💡 Notice: Semantic search finds relevant documents even when")
    print("   exact keywords don't match! It understands meaning.\n")

    # Cleanup
    client.delete_collection("semantic_search")


async def main():
    print("=" * 70)
    print("Semantic Kernel - Step 7: Memory & RAG")
    print("=" * 70)
    print("\nMemory & RAG allow AI to:")
    print("  • Remember information across conversations")
    print("  • Search documents by meaning (semantic search)")
    print("  • Answer questions using your documents (RAG)")
    print("  • Build knowledge bases and Q&A systems")
    print("\n" + "=" * 70 + "\n")

    await example1_basic_memory_storage()
    print()

    await example2_document_qa_with_rag()
    print()

    await example3_conversational_memory()
    print()

    await example4_semantic_search_demo()

    print("=" * 70)
    print("✅ Step 7 Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Vector databases store text as embeddings (numerical vectors)")
    print("  • Semantic search finds similar content by meaning, not keywords")
    print("  • RAG pattern: Retrieve → Augment → Generate")
    print("  • Memory enables AI to remember facts across conversations")
    print("  • ChromaDB is a simple, powerful vector database")
    print("\nRAG Pattern Steps:")
    print("  1. RETRIEVE: Search vector DB for relevant documents")
    print("  2. AUGMENT: Add retrieved context to the prompt")
    print("  3. GENERATE: AI answers using the context")
    print("\nWhen to Use Memory & RAG:")
    print("  ✅ Document Q&A systems")
    print("  ✅ Knowledge bases and wikis")
    print("  ✅ Customer support with product documentation")
    print("  ✅ Chatbots that remember user preferences")
    print("  ✅ Research assistants that search papers")


if __name__ == "__main__":
    asyncio.run(main())

