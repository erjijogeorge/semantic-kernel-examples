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

## Project Structure

```
semantic-kernel-app/
├── examples/
│   ├── step1_basic_chat.py
│   ├── step2_prompt_templates.py
│   ├── step3_chat_history.py
│   ├── step4_native_functions.py
│   └── plugins/
│       ├── __init__.py
│       ├── math_plugin.py
│       ├── weather_plugin.py
│       └── database_plugin.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Next Steps

- Step 5: Function Calling & Auto-Invocation
- Step 6: Streaming Responses
- Step 7: Memory & RAG
- Step 8: Planners & Agents

## Resources

- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [GitHub Repository](https://github.com/microsoft/semantic-kernel)
- [Python Samples](https://github.com/microsoft/semantic-kernel/tree/main/python)

## License

MIT

