import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

load_dotenv()


async def main():
    kernel = Kernel()
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    service_id = "chat"
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            api_key=api_key,
            endpoint=endpoint,
            deployment_name=deployment_name,
        )
    )
    
    # Simple template with variables
    simple_template = """
    You are a helpful assistant.
    
    Question: {{$question}}
    
    Provide a clear answer.
    """
    
    simple_function = kernel.add_function(
        prompt=simple_template,
        function_name="simple_qa",
        plugin_name="QAPlugin",
    )
    
    result = await kernel.invoke(
        simple_function,
        question="What is machine learning?"
    )
    print(f"Simple Template:\n{result}\n")
    
    # Template with multiple variables
    multi_var_template = """
    You are a {{$role}}.
    
    Topic: {{$topic}}
    Audience: {{$audience}}
    
    Explain the topic appropriately for the audience.
    """
    
    multi_var_function = kernel.add_function(
        prompt=multi_var_template,
        function_name="explain",
        plugin_name="ExplainPlugin",
    )
    
    result = await kernel.invoke(
        multi_var_function,
        role="teacher",
        topic="neural networks",
        audience="high school students"
    )
    print(f"Multi-Variable Template:\n{result}\n")
    
    # Template with execution settings
    settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        temperature=0.7,
        max_tokens=150,
        top_p=0.9,
    )
    
    creative_template = """
    Write a creative {{$style}} about {{$subject}}.
    Keep it to 2-3 sentences.
    """
    
    creative_function = kernel.add_function(
        prompt=creative_template,
        function_name="creative_writer",
        plugin_name="WriterPlugin",
    )
    
    result = await kernel.invoke(
        creative_function,
        style="haiku",
        subject="coding",
        settings=settings
    )
    print(f"With Execution Settings:\n{result}\n")


if __name__ == "__main__":
    asyncio.run(main())

