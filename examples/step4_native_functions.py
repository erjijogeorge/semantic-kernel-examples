"""
Step 4: Native Functions (Plugins)

This example demonstrates how to create and use custom Python functions (plugins)
that AI can call to perform specific tasks.

Concepts:
- Creating native functions with @kernel_function decorator
- Building plugins with multiple related functions
- Combining AI reasoning with traditional code logic
- Type annotations for better AI understanding
- Function descriptions for AI to know when to use them

Plugins demonstrated:
1. MathPlugin - Mathematical operations
2. WeatherPlugin - Simulated API calls
3. DatabasePlugin - Simulated database queries
"""

import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory

# Import our custom plugins
from plugins import MathPlugin, WeatherPlugin, DatabasePlugin

load_dotenv()


async def main():
    # Initialize kernel
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
    
    # Add plugins to kernel
    kernel.add_plugin(MathPlugin(), plugin_name="Math")
    kernel.add_plugin(WeatherPlugin(), plugin_name="Weather")
    kernel.add_plugin(DatabasePlugin(), plugin_name="Database")
    
    print("=" * 70)
    print("Semantic Kernel - Step 4: Native Functions (Plugins)")
    print("=" * 70)
    print("\nPlugins loaded:")
    print("  • MathPlugin - Mathematical operations")
    print("  • WeatherPlugin - Weather information")
    print("  • DatabasePlugin - Data queries")
    print("\n" + "=" * 70 + "\n")
    
    # Configure execution settings to enable auto function calling
    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.7,
        max_tokens=1000,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )
    
    # Create chat history
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a helpful AI assistant with access to math, weather, and database functions. "
        "Use these functions when needed to answer user questions accurately. "
        "Always explain what you're doing and show your work."
    )
    
    # Example 1: Math operations
    print("📊 Example 1: Math Operations")
    print("-" * 70)
    
    chat_history.add_user_message(
        "I need to calculate: (15 + 25) × 3, then find the square root of the result."
    )
    
    response = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel,
    )
    
    chat_history.add_assistant_message(str(response))
    print(f"User: I need to calculate: (15 + 25) × 3, then find the square root of the result.")
    print(f"Assistant: {response}\n")
    
    # Example 2: Weather queries
    print("🌤️  Example 2: Weather Information")
    print("-" * 70)
    
    chat_history.add_user_message(
        "What's the weather like in London? Should I bring an umbrella?"
    )
    
    response = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel,
    )
    
    chat_history.add_assistant_message(str(response))
    print(f"User: What's the weather like in London? Should I bring an umbrella?")
    print(f"Assistant: {response}\n")
    
    # Example 3: Database queries
    print("💾 Example 3: Database Operations")
    print("-" * 70)
    
    chat_history.add_user_message(
        "Find user Alice and show me her orders."
    )
    
    response = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel,
    )
    
    chat_history.add_assistant_message(str(response))
    print(f"User: Find user Alice and show me her orders.")
    print(f"Assistant: {response}\n")
    
    # Example 4: Complex multi-step query
    print("🔄 Example 4: Multi-Step Complex Query")
    print("-" * 70)
    
    chat_history.add_user_message(
        "Search for electronics products, calculate the total value of laptops in stock, "
        "and tell me what percentage of total inventory value that represents."
    )
    
    response = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel,
    )
    
    chat_history.add_assistant_message(str(response))
    print(f"User: Search for electronics products, calculate the total value...")
    print(f"Assistant: {response}\n")
    
    print("=" * 70)
    print("✅ Step 4 Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Native functions let AI call your Python code")
    print("  • @kernel_function decorator makes functions discoverable")
    print("  • Type annotations help AI understand parameters")
    print("  • Descriptions guide AI on when to use functions")
    print("  • AI can chain multiple function calls automatically")


if __name__ == "__main__":
    asyncio.run(main())

