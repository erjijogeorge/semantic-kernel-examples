import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

load_dotenv()


async def main():
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
    
    settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.7,
        max_tokens=500,
    )
    
    # Create chat history
    chat_history = ChatHistory()
    
    # Add system message to set behavior
    chat_history.add_system_message(
        "You are a helpful AI assistant who explains technical concepts clearly and concisely."
    )
    
    # First user message
    chat_history.add_user_message("What is a REST API?")
    
    response = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=settings,
    )
    
    chat_history.add_assistant_message(str(response))
    print(f"User: What is a REST API?")
    print(f"Assistant: {response}\n")
    
    # Follow-up question (uses context from previous messages)
    chat_history.add_user_message("Can you give me a simple example?")
    
    response = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=settings,
    )
    
    chat_history.add_assistant_message(str(response))
    print(f"User: Can you give me a simple example?")
    print(f"Assistant: {response}\n")
    
    # Another follow-up
    chat_history.add_user_message("What HTTP methods are commonly used?")
    
    response = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=settings,
    )
    
    chat_history.add_assistant_message(str(response))
    print(f"User: What HTTP methods are commonly used?")
    print(f"Assistant: {response}\n")
    
    # Show full conversation history
    print("=" * 70)
    print("Full Conversation History:")
    print("=" * 70)
    for i, message in enumerate(chat_history.messages):
        print(f"{i+1}. {message.role}: {message.content}")


if __name__ == "__main__":
    asyncio.run(main())

