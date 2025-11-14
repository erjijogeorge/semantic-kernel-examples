import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

load_dotenv()


async def main():
    kernel = Kernel()

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    if not api_key or not endpoint or not deployment_name:
        print("Error: Missing Azure OpenAI config in .env file")
        return

    service_id = "chat"
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            api_key=api_key,
            endpoint=endpoint,
            deployment_name=deployment_name,
        )
    )

    chat_function = kernel.add_function(
        prompt="What is AI/ML? Answer in 2 to 4 sentences.",
        function_name="chat",
        plugin_name="ChatPlugin",
    )

    result = await kernel.invoke(chat_function)
    print(f"Response: {result}")


if __name__ == "__main__":
    asyncio.run(main())

