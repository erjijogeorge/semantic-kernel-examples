# Semantic Kernel Examples

Learn Semantic Kernel step-by-step with practical Python examples using Azure OpenAI.

## What is Semantic Kernel?

Semantic Kernel is an open-source SDK by Microsoft that integrates Large Language Models (LLMs) with conventional programming. It enables you to build AI agents and applications that combine AI capabilities with your code.

## Prerequisites

- Python 3.12+
- Azure OpenAI account with a deployed model
- Basic Python knowledge

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/erjijogeorge/semantic-kernel-examples.git
cd semantic-kernel-examples
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and add your Azure OpenAI credentials:

```
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
```

Get these from Azure Portal → Your OpenAI Resource → Keys and Endpoint

## Examples

### Step 1: Basic Chat Completion

Learn the fundamentals of Semantic Kernel.

```bash
python examples/step1_basic_chat.py
```

**Concepts:**
- Initialize Kernel
- Add Azure OpenAI service
- Create and invoke a simple chat function

### Step 2: Prompt Templates

Use variables to make prompts dynamic and reusable.

```bash
python examples/step2_prompt_templates.py
```

**Concepts:**
- Template variables (`{{$variable}}`)
- Passing arguments to functions
- Execution settings (temperature, max_tokens)

### Step 3: Chat History & Conversations

Build multi-turn conversations with context.

```bash
python examples/step3_chat_history.py
```

**Concepts:**
- ChatHistory for conversation management
- System/user/assistant messages
- Context awareness across messages

### Step 4: Native Functions (Plugins)

Create custom Python functions that AI can call automatically.

```bash
python examples/step4_native_functions.py
```

**Concepts:**
- Creating native functions with `@kernel_function` decorator
- Building plugins with multiple related functions
- Type annotations for better AI understanding
- Function descriptions to guide AI
- Auto function calling with `FunctionChoiceBehavior`
- Combining AI reasoning with traditional code

**Plugins Included:**
- **MathPlugin**: Mathematical operations (add, multiply, power, percentage, etc.)
- **WeatherPlugin**: Simulated weather API calls
- **DatabasePlugin**: Simulated database queries (users, products, orders)

**What You'll Learn:**
- How to make your Python functions callable by AI
- How AI automatically chooses which functions to call
- How to chain multiple function calls together
- Real-world patterns: API calls, database queries, calculations

### Step 5: Semantic Functions

Create AI-powered functions defined by prompts (not code).

```bash
python examples/step5_semantic_functions.py
```

**Concepts:**
- Creating semantic functions from prompt templates
- Organizing functions with `config.json` and `skprompt.txt`
- Reusable AI capabilities
- Chaining semantic functions together
- Difference between native and semantic functions

**Semantic Functions Included:**
- **TextAnalysis**: Summarize, SentimentAnalysis, ExtractKeywords
- **Translation**: Translate to any language
- **Creative**: WritePoem, GenerateIdeas

**What You'll Learn:**
- How to create AI functions using prompts instead of code
- How to organize semantic functions in folders
- How to configure function parameters and settings
- How to chain multiple semantic functions into pipelines
- When to use semantic vs native functions

**Key Difference:**
- **Native Functions (Step 4)**: Python code that AI calls
- **Semantic Functions (Step 5)**: AI prompts that become reusable functions

### Step 6: Streaming Responses

Get real-time, token-by-token responses from the AI.

```bash
python examples/step6_streaming.py
```

**Concepts:**
- Streaming vs non-streaming responses
- Real-time token-by-token output
- Streaming with chat history
- Streaming with function calling
- Better user experience with immediate feedback

**Examples Included:**
- **Basic Streaming**: Simple streaming response
- **Comparison**: Streaming vs non-streaming side-by-side
- **Chat History**: Multi-turn conversations with streaming
- **Function Calling**: Streaming with auto function calling
- **Interactive Chat**: Simulated interactive chat application

**What You'll Learn:**
- How to stream responses in real-time
- When to use streaming vs non-streaming
- How to combine streaming with chat history
- How to stream with function calling enabled
- Building responsive chat applications

**Key Difference:**
- **Non-Streaming**: Wait → Get complete response → Display
- **Streaming**: Get tokens → Display immediately → Continue

## Project Structure

```
semantic-kernel-app/
├── examples/
│   ├── step1_basic_chat.py
│   ├── step2_prompt_templates.py
│   ├── step3_chat_history.py
│   ├── step4_native_functions.py
│   ├── step5_semantic_functions.py
│   ├── step6_streaming.py
│   ├── plugins/                          # Native function plugins
│   │   ├── __init__.py
│   │   ├── math_plugin.py
│   │   ├── weather_plugin.py
│   │   └── database_plugin.py
│   └── semantic_functions/               # Semantic function definitions
│       ├── TextAnalysis/
│       │   ├── Summarize/
│       │   │   ├── config.json
│       │   │   └── skprompt.txt
│       │   ├── SentimentAnalysis/
│       │   │   ├── config.json
│       │   │   └── skprompt.txt
│       │   └── ExtractKeywords/
│       │       ├── config.json
│       │       └── skprompt.txt
│       ├── Translation/
│       │   └── Translate/
│       │       ├── config.json
│       │       └── skprompt.txt
│       └── Creative/
│           ├── WritePoem/
│           │   ├── config.json
│           │   └── skprompt.txt
│           └── GenerateIdeas/
│               ├── config.json
│               └── skprompt.txt
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Next Steps

- Step 7: Memory & RAG (Retrieval Augmented Generation)
- Step 8: Planners & Autonomous Agents
- Step 9: Interactive Chat Application with All Features

## Resources

- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [GitHub Repository](https://github.com/microsoft/semantic-kernel)
- [Python Samples](https://github.com/microsoft/semantic-kernel/tree/main/python)

## License

MIT

