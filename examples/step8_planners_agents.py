"""
Step 8: Planners & Autonomous Agents

This example demonstrates how to build AI agents that can plan and execute
multi-step tasks autonomously.

Concepts:
- Autonomous agents that break down complex tasks
- Multi-step planning and execution
- Function calling with reasoning
- Agent-driven workflows
- Task decomposition and orchestration

Key Components:
- ChatCompletionAgent: AI that plans and executes tasks
- Auto function calling: AI selects and chains functions
- Multi-step reasoning: AI breaks down complex goals
- Autonomous execution: AI completes tasks without manual orchestration
"""

import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory

# Import plugins
from plugins import MathPlugin, WeatherPlugin, DatabasePlugin

load_dotenv()


async def example1_simple_autonomous_task():
    """Example 1: Simple autonomous task - AI plans and executes"""
    print("🤖 Example 1: Simple Autonomous Task")
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
    
    # Add plugins
    kernel.add_plugin(MathPlugin(), plugin_name="Math")
    kernel.add_plugin(WeatherPlugin(), plugin_name="Weather")
    
    # Enable auto function calling - AI will plan and execute
    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.0,  # Lower temperature for more deterministic planning
        max_tokens=1000,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )
    
    # Give the AI a complex task
    task = """
    I need to calculate the total cost of a trip:
    - Flight costs $450
    - Hotel costs $120 per night for 3 nights
    - Daily expenses are $80 per day for 4 days
    
    Calculate the total cost and then add 15% for unexpected expenses.
    """
    
    print(f"📋 Task: {task.strip()}\n")
    print("🧠 AI is planning and executing...\n")
    
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a helpful assistant that can perform calculations. "
        "Break down complex problems into steps and use available functions."
    )
    chat_history.add_user_message(task)
    
    # AI autonomously plans and executes
    result = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel
    )
    
    print(f"\n✅ Final Answer: {result}\n")


async def example2_multi_step_research_task():
    """Example 2: Multi-step research task with multiple data sources"""
    print("🔬 Example 2: Multi-Step Research Task")
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
    
    # Add plugins
    kernel.add_plugin(WeatherPlugin(), plugin_name="Weather")
    kernel.add_plugin(DatabasePlugin(), plugin_name="Database")
    
    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.0,
        max_tokens=1500,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )
    
    task = """
    I'm planning a business trip. Help me with the following:
    1. Check the weather in New York and London
    2. Find user Alice in the database and get her contact info
    3. Search for products in the 'Electronics' category
    4. Based on the weather, recommend which city would be better for outdoor meetings
    """
    
    print(f"📋 Task: {task.strip()}\n")
    print("🧠 AI Agent is working...\n")
    
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a helpful research assistant. Use available tools to gather information "
        "and provide comprehensive answers. Execute tasks step by step."
    )
    chat_history.add_user_message(task)
    
    result = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel
    )
    
    print(f"\n✅ Research Complete:\n{result}\n")


async def example3_conversational_agent():
    """Example 3: Conversational agent that maintains context and plans"""
    print("💬 Example 3: Conversational Agent with Planning")
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

    # Add all plugins
    kernel.add_plugin(MathPlugin(), plugin_name="Math")
    kernel.add_plugin(WeatherPlugin(), plugin_name="Weather")
    kernel.add_plugin(DatabasePlugin(), plugin_name="Database")

    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.3,
        max_tokens=1000,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )

    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are an intelligent assistant that can help with calculations, weather, "
        "and database queries. You maintain context across the conversation and "
        "proactively use tools when needed."
    )

    print("Starting conversation with autonomous agent...\n")

    # Simulate a multi-turn conversation
    user_messages = [
        "What's the weather like in Tokyo?",
        "How does that compare to Paris?",
        "If the temperature difference is more than 10 degrees, calculate what percentage warmer one is than the other.",
        "Great! Now find all users in the database who are engineers.",
    ]

    for i, user_msg in enumerate(user_messages, 1):
        print(f"👤 User (Turn {i}): {user_msg}")
        chat_history.add_user_message(user_msg)

        # Agent autonomously decides what to do
        response = await chat_service.get_chat_message_content(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel
        )

        assistant_msg = str(response)
        chat_history.add_assistant_message(assistant_msg)
        print(f"🤖 Agent: {assistant_msg}\n")


async def example4_goal_oriented_agent():
    """Example 4: Goal-oriented agent that works towards a specific objective"""
    print("🎯 Example 4: Goal-Oriented Agent")
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

    # Add plugins
    kernel.add_plugin(MathPlugin(), plugin_name="Math")
    kernel.add_plugin(DatabasePlugin(), plugin_name="Database")

    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.0,
        max_tokens=2000,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )

    # Complex goal that requires planning
    goal = """
    GOAL: Analyze our customer database and calculate business metrics.

    Tasks:
    1. Get all users from the database
    2. Count how many users we have
    3. Get all products
    4. Calculate the average price of all products
    5. Get all orders
    6. Calculate the total revenue (sum of all order amounts)
    7. Provide a summary report with all these metrics

    Execute these tasks systematically and provide a comprehensive report.
    """

    print(f"🎯 Goal:\n{goal}\n")
    print("🤖 Agent is planning and executing tasks...\n")

    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a business analyst agent. You systematically execute tasks, "
        "use available tools, and provide detailed reports. "
        "Work through tasks step by step and show your reasoning."
    )
    chat_history.add_user_message(goal)

    result = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel
    )

    print(f"📊 Business Analysis Report:\n{result}\n")


async def example5_adaptive_agent():
    """Example 5: Adaptive agent that adjusts strategy based on results"""
    print("🔄 Example 5: Adaptive Agent")
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

    # Add plugins
    kernel.add_plugin(MathPlugin(), plugin_name="Math")
    kernel.add_plugin(WeatherPlugin(), plugin_name="Weather")

    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.2,
        max_tokens=1500,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )

    task = """
    I'm organizing an outdoor event and need to make decisions:

    1. Check the weather in San Francisco, New York, and Miami
    2. For each city, determine if it's suitable for an outdoor event (temperature between 65-80°F and not rainy)
    3. Calculate the average temperature across all suitable cities
    4. Recommend the best city based on weather conditions
    5. If no cities are suitable, suggest postponing the event

    Make data-driven decisions and explain your reasoning.
    """

    print(f"📋 Task: {task.strip()}\n")
    print("🤖 Adaptive Agent is analyzing and deciding...\n")

    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are an event planning agent. You gather data, analyze it, "
        "and make informed recommendations. Adapt your strategy based on "
        "the information you discover."
    )
    chat_history.add_user_message(task)

    result = await chat_service.get_chat_message_content(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel
    )

    print(f"📍 Event Planning Recommendation:\n{result}\n")


async def main():
    print("=" * 70)
    print("Semantic Kernel - Step 8: Planners & Autonomous Agents")
    print("=" * 70)
    print("\nAutonomous Agents can:")
    print("  • Break down complex tasks into steps")
    print("  • Plan and execute multi-step workflows")
    print("  • Use tools and functions autonomously")
    print("  • Adapt strategies based on results")
    print("  • Maintain context across conversations")
    print("\n" + "=" * 70 + "\n")

    await example1_simple_autonomous_task()
    print()

    await example2_multi_step_research_task()
    print()

    await example3_conversational_agent()
    print()

    await example4_goal_oriented_agent()
    print()

    await example5_adaptive_agent()

    print("=" * 70)
    print("✅ Step 8 Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Autonomous agents plan and execute tasks without manual orchestration")
    print("  • FunctionChoiceBehavior.Auto() enables AI to select and chain functions")
    print("  • Lower temperature (0.0-0.3) makes planning more deterministic")
    print("  • Agents can maintain context across multi-turn conversations")
    print("  • System messages guide agent behavior and strategy")
    print("\nAgent Capabilities:")
    print("  ✅ Task decomposition (breaking complex goals into steps)")
    print("  ✅ Multi-step reasoning (planning sequences of actions)")
    print("  ✅ Tool selection (choosing the right functions)")
    print("  ✅ Adaptive behavior (adjusting based on results)")
    print("  ✅ Conversational memory (maintaining context)")
    print("\nWhen to Use Autonomous Agents:")
    print("  ✅ Complex workflows with multiple steps")
    print("  ✅ Tasks requiring data from multiple sources")
    print("  ✅ Decision-making based on gathered information")
    print("  ✅ Interactive assistants that need to plan")
    print("  ✅ Business process automation")
    print("\n💡 Pro Tip: Combine agents with Memory & RAG (Step 7) for")
    print("   powerful knowledge-based autonomous systems!")


if __name__ == "__main__":
    asyncio.run(main())

