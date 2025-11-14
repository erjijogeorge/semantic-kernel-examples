"""
Step 5: Semantic Functions

This example demonstrates how to create and use semantic functions - AI-powered
functions defined by prompts rather than code.

Concepts:
- Creating semantic functions from prompt templates
- Organizing functions in a structured folder hierarchy
- Using config.json for function settings
- Reusable AI capabilities
- Chaining semantic functions together
- Difference between native functions (Step 4) and semantic functions (Step 5)

Key Difference:
- Native Functions (Step 4): Python code that AI calls
- Semantic Functions (Step 5): AI prompts that become reusable functions
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments

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
    
    # Get the path to semantic functions folder
    script_dir = Path(__file__).parent
    plugins_directory = script_dir / "semantic_functions"
    
    # Import semantic functions from folders
    # Each folder becomes a plugin, each subfolder becomes a function
    text_analysis_plugin = kernel.add_plugin(
        parent_directory=str(plugins_directory),
        plugin_name="TextAnalysis"
    )
    
    translation_plugin = kernel.add_plugin(
        parent_directory=str(plugins_directory),
        plugin_name="Translation"
    )
    
    creative_plugin = kernel.add_plugin(
        parent_directory=str(plugins_directory),
        plugin_name="Creative"
    )
    
    print("=" * 70)
    print("Semantic Kernel - Step 5: Semantic Functions")
    print("=" * 70)
    print("\nSemantic Functions loaded:")
    print("  📊 TextAnalysis: Summarize, SentimentAnalysis, ExtractKeywords")
    print("  🌍 Translation: Translate")
    print("  🎨 Creative: WritePoem, GenerateIdeas")
    print("\n" + "=" * 70 + "\n")
    
    # Sample text for analysis
    sample_text = """
    Artificial Intelligence is revolutionizing the way we work and live. 
    Machine learning algorithms can now process vast amounts of data, 
    identify patterns, and make predictions with remarkable accuracy. 
    From healthcare to finance, AI is transforming industries and creating 
    new opportunities. However, it also raises important questions about 
    privacy, ethics, and the future of work. As we continue to develop 
    these powerful technologies, we must ensure they benefit all of humanity.
    """
    
    # Example 1: Text Summarization
    print("📝 Example 1: Text Summarization")
    print("-" * 70)
    
    result = await kernel.invoke(
        text_analysis_plugin["Summarize"],
        KernelArguments(input=sample_text, max_words="30")
    )
    
    print(f"Original text length: {len(sample_text.split())} words")
    print(f"\nSummary (30 words max):\n{result}\n")
    
    # Example 2: Sentiment Analysis
    print("😊 Example 2: Sentiment Analysis")
    print("-" * 70)
    
    result = await kernel.invoke(
        text_analysis_plugin["SentimentAnalysis"],
        KernelArguments(input=sample_text)
    )
    
    print(f"Text: {sample_text.strip()[:100]}...")
    print(f"\nSentiment Analysis:\n{result}\n")
    
    # Example 3: Extract Keywords
    print("🔑 Example 3: Extract Keywords")
    print("-" * 70)
    
    result = await kernel.invoke(
        text_analysis_plugin["ExtractKeywords"],
        KernelArguments(input=sample_text, count="7")
    )
    
    print(f"Extracted Keywords:\n{result}\n")
    
    # Example 4: Translation
    print("🌍 Example 4: Translation")
    print("-" * 70)
    
    english_text = "Hello! How are you today? I hope you're having a wonderful day."
    
    result = await kernel.invoke(
        translation_plugin["Translate"],
        KernelArguments(input=english_text, target_language="French")
    )
    
    print(f"English: {english_text}")
    print(f"French: {result}\n")
    
    # Example 5: Creative Writing - Poem
    print("🎨 Example 5: Creative Writing - Poem")
    print("-" * 70)
    
    result = await kernel.invoke(
        creative_plugin["WritePoem"],
        KernelArguments(input="artificial intelligence", style="haiku", length="3")
    )
    
    print(f"Haiku about AI:\n{result}\n")
    
    # Example 6: Generate Ideas
    print("💡 Example 6: Generate Creative Ideas")
    print("-" * 70)
    
    result = await kernel.invoke(
        creative_plugin["GenerateIdeas"],
        KernelArguments(input="improving team productivity", count="3")
    )
    
    print(f"Ideas for improving team productivity:\n{result}\n")
    
    # Example 7: Chaining Semantic Functions
    print("🔗 Example 7: Chaining Semantic Functions")
    print("-" * 70)
    print("Pipeline: Extract Keywords → Generate Ideas → Translate to Spanish\n")

    # Step 1: Extract keywords from sample text
    keywords_result = await kernel.invoke(
        text_analysis_plugin["ExtractKeywords"],
        KernelArguments(input=sample_text, count="3")
    )
    print(f"Step 1 - Keywords extracted: {keywords_result}")

    # Step 2: Generate ideas based on those keywords
    ideas_result = await kernel.invoke(
        creative_plugin["GenerateIdeas"],
        KernelArguments(input=str(keywords_result), count="2")
    )
    print(f"\nStep 2 - Ideas generated:\n{ideas_result}")

    # Step 3: Translate the ideas to Spanish
    translation_result = await kernel.invoke(
        translation_plugin["Translate"],
        KernelArguments(input=str(ideas_result), target_language="Spanish")
    )
    print(f"\nStep 3 - Translated to Spanish:\n{translation_result}\n")

    # Example 8: Analyze → Summarize → Translate Pipeline
    print("🔗 Example 8: Multi-Step Analysis Pipeline")
    print("-" * 70)

    review_text = """
    This product exceeded all my expectations! The quality is outstanding,
    and the customer service was incredibly helpful. I had a small issue
    with shipping, but they resolved it immediately. The price is a bit
    high, but it's worth every penny. I've already recommended it to
    three friends. Definitely buying again!
    """

    print("Pipeline: Sentiment Analysis → Summarize → Translate to German\n")

    # Step 1: Analyze sentiment
    sentiment = await kernel.invoke(
        text_analysis_plugin["SentimentAnalysis"],
        KernelArguments(input=review_text)
    )
    print(f"Step 1 - Sentiment: {sentiment}")

    # Step 2: Summarize the review
    summary = await kernel.invoke(
        text_analysis_plugin["Summarize"],
        KernelArguments(input=review_text, max_words="20")
    )
    print(f"\nStep 2 - Summary: {summary}")

    # Step 3: Translate summary to German
    german_summary = await kernel.invoke(
        translation_plugin["Translate"],
        KernelArguments(input=str(summary), target_language="German")
    )
    print(f"\nStep 3 - German Translation: {german_summary}\n")

    print("=" * 70)
    print("✅ Step 5 Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Semantic functions are AI-powered, defined by prompts")
    print("  • Organized in folders: Plugin/Function/config.json + skprompt.txt")
    print("  • Reusable across different applications")
    print("  • Can be chained together for complex workflows")
    print("  • Different from native functions (code vs prompts)")
    print("\nNative vs Semantic Functions:")
    print("  Native (Step 4):   Python code → AI calls it")
    print("  Semantic (Step 5): AI prompt → Becomes a function")


if __name__ == "__main__":
    asyncio.run(main())

