"""
Step 6: Streaming Responses

This example demonstrates how to stream responses from the AI in real-time,
displaying tokens as they're generated instead of waiting for the complete response.

Concepts:
- Streaming vs non-streaming responses
- Real-time token-by-token output
- Streaming with chat history
- Streaming with function calling
- Better user experience with immediate feedback

Key Difference:
- Non-Streaming: Wait → Get complete response → Display
- Streaming: Get tokens → Display immediately → Continue
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Import plugins from Step 4
sys.path.append(os.path.join(os.path.dirname(__file__), 'plugins'))
from math_plugin import MathPlugin

load_dotenv()


async def example1_basic_streaming():
    """Example 1: Basic streaming response"""
    print("📡 Example 1: Basic Streaming Response")
    print("-" * 70)
    
    kernel = Kernel()
    
    # Setup Azure OpenAI
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
    
    # Create execution settings
    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.7,
        max_tokens=500,
    )
    
    prompt = "Write a short story (3 paragraphs) about a robot learning to paint."
    
    print(f"Prompt: {prompt}\n")
    print("Streaming response:")
    print("-" * 70)
    
    # Stream the response
    full_response = ""
    async for chunk in kernel.invoke_prompt_stream(
        prompt=prompt,
        settings=execution_settings
    ):
        if chunk:
            content = str(chunk[0])
            print(content, end="", flush=True)
            full_response += content
    
    print("\n" + "-" * 70)
    print(f"✅ Streamed {len(full_response)} characters\n")


async def example2_streaming_vs_non_streaming():
    """Example 2: Compare streaming vs non-streaming"""
    print("⚡ Example 2: Streaming vs Non-Streaming Comparison")
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
    
    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.7,
        max_tokens=300,
    )
    
    prompt = "Explain quantum computing in simple terms (2 paragraphs)."
    
    # Non-streaming (wait for complete response)
    print("🔵 NON-STREAMING (wait for complete response):")
    print("-" * 70)
    import time
    start = time.time()
    
    result = await kernel.invoke_prompt(
        prompt=prompt,
        settings=execution_settings
    )
    
    elapsed = time.time() - start
    print(f"[Waited {elapsed:.2f}s... then got complete response]\n")
    print(result)
    print()
    
    # Streaming (see tokens as they arrive)
    print("🟢 STREAMING (see tokens as they arrive):")
    print("-" * 70)
    start = time.time()
    
    async for chunk in kernel.invoke_prompt_stream(
        prompt=prompt,
        settings=execution_settings
    ):
        if chunk:
            print(str(chunk[0]), end="", flush=True)
    
    elapsed = time.time() - start
    print(f"\n[Completed in {elapsed:.2f}s with real-time display]\n")


async def example3_streaming_with_chat_history():
    """Example 3: Streaming with chat history"""
    print("💬 Example 3: Streaming with Chat History")
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

    # Create chat history
    chat_history = ChatHistory()
    chat_history.add_system_message("You are a helpful AI assistant. Keep responses concise.")

    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.7,
        max_tokens=300,
    )

    # Multi-turn conversation with streaming
    conversations = [
        "What is Python?",
        "What are its main advantages?",
        "Can you give me a simple code example?"
    ]

    for user_message in conversations:
        chat_history.add_user_message(user_message)

        print(f"\n👤 User: {user_message}")
        print(f"🤖 Assistant: ", end="", flush=True)

        # Stream the response
        full_response = ""
        async for chunk in chat_service.get_streaming_chat_message_contents(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel
        ):
            if chunk:
                content = str(chunk[0])
                print(content, end="", flush=True)
                full_response += content

        # Add assistant's response to history
        chat_history.add_assistant_message(full_response)
        print()  # New line after response

    print()


async def example4_streaming_with_functions():
    """Example 4: Streaming with function calling"""
    print("🔧 Example 4: Streaming with Function Calling")
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

    # Add Math plugin
    kernel.add_plugin(MathPlugin(), plugin_name="Math")

    # Enable auto function calling
    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.7,
        max_tokens=500,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )

    chat_history = ChatHistory()
    chat_history.add_system_message("You are a helpful math assistant.")

    user_message = "What's 15 squared, then add 25, then divide by 5? Explain each step."
    chat_history.add_user_message(user_message)

    print(f"👤 User: {user_message}")
    print(f"🤖 Assistant: ", end="", flush=True)

    # Stream response with function calling
    full_response = ""
    async for chunk in chat_service.get_streaming_chat_message_contents(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel
    ):
        if chunk:
            content = str(chunk[0])
            if content:  # Only print non-empty content
                print(content, end="", flush=True)
                full_response += content

    print("\n")


async def example5_interactive_streaming_chat():
    """Example 5: Interactive streaming chat (simulated)"""
    print("🎮 Example 5: Interactive Streaming Chat (Simulated)")
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

    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a friendly AI assistant. Be helpful, concise, and engaging."
    )

    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.8,
        max_tokens=400,
    )

    # Simulated user inputs (in real app, these would come from user input)
    simulated_inputs = [
        "Hello! What can you help me with?",
        "Tell me an interesting fact about space.",
        "Thanks! Goodbye!"
    ]

    print("Starting interactive chat (simulated)...\n")

    for user_input in simulated_inputs:
        print(f"👤 You: {user_input}")

        chat_history.add_user_message(user_input)

        print(f"🤖 AI: ", end="", flush=True)

        full_response = ""
        async for chunk in chat_service.get_streaming_chat_message_contents(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel
        ):
            if chunk:
                content = str(chunk[0])
                print(content, end="", flush=True)
                full_response += content

        chat_history.add_assistant_message(full_response)
        print("\n")

    print("Chat ended.\n")


async def main():
    print("=" * 70)
    print("Semantic Kernel - Step 6: Streaming Responses")
    print("=" * 70)
    print("\nStreaming allows you to see AI responses in real-time,")
    print("token-by-token, instead of waiting for the complete response.")
    print("\n" + "=" * 70 + "\n")

    # Run all examples
    await example1_basic_streaming()
    print()

    await example2_streaming_vs_non_streaming()
    print()

    await example3_streaming_with_chat_history()
    print()

    await example4_streaming_with_functions()
    print()

    await example5_interactive_streaming_chat()

    print("=" * 70)
    print("✅ Step 6 Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Streaming provides real-time token-by-token output")
    print("  • Better user experience - see progress immediately")
    print("  • Works with chat history and function calling")
    print("  • Essential for interactive chat applications")
    print("  • Use invoke_prompt_stream() or get_streaming_chat_message_contents()")
    print("\nWhen to Use Streaming:")
    print("  ✅ Chat applications and chatbots")
    print("  ✅ Long-form content generation")
    print("  ✅ Interactive user experiences")
    print("  ✅ When you want to show progress")
    print("\nWhen to Use Non-Streaming:")
    print("  ✅ Batch processing")
    print("  ✅ When you need the complete response first")
    print("  ✅ Background tasks without UI")


if __name__ == "__main__":
    asyncio.run(main())

