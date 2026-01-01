# CLAUDE.MD - Project Context for AI Assistants

## Project Overview

This is a **Basic Chat Voice Agent** template built on the Cartesia Line platform. It implements a conversational voice agent that combines:
- **Cartesia Sonic** for text-to-speech (TTS)
- **Cartesia Ink** for speech-to-text (STT)
- **Google Gemini 2.5 Flash** for language modeling

The agent is designed for customer service, personal assistants, educational tutoring, and business receptionists.

## Architecture

### Core Components

1. **main.py** - Entry point and event routing
   - Initializes the VoiceAgentApp
   - Handles new call setup via `handle_new_call()`
   - Creates the conversation node and bridge
   - Routes events between user input and the ChatNode
   - Sends initial greeting message

2. **chat_node.py** - ChatNode class (extends ReasoningNode)
   - Voice-optimized ReasoningNode implementation using Gemini streaming
   - Uses template method pattern from ReasoningNode
   - Implements `process_context()` for Gemini streaming
   - Handles tool calls (specifically the EndCallTool)
   - Includes interruption support via `stop_generation_event`
   - Provides canned responses when Gemini API key is missing

3. **config.py** - Configuration and system prompt
   - Defines DEFAULT_MODEL_ID (gemini-2.5-flash)
   - Sets DEFAULT_TEMPERATURE (0.7)
   - Contains SYSTEM_PROMPT with agent personality and guidelines

### Event Flow

```
User speaks → UserStoppedSpeaking event
    ↓
ChatNode.generate() called
    ↓
process_context() streams from Gemini
    ↓
AgentResponse events yielded
    ↓
Text-to-speech via Cartesia Sonic
    ↓
User hears response
```

Interruptions are handled via UserStartedSpeaking events that can interrupt ongoing generation.

## Key Implementation Details

### Conversation Management

- Uses **ConversationContext** to track conversation history
- Maximum context length: 100 turns (configurable via `max_context_length`)
- Messages are converted to Gemini format via `convert_messages_to_gemini()`
- Context events include user transcriptions and agent responses

### Tool Integration

The agent implements the **EndCallTool** which allows the LLM to end conversations:
- Tool is registered in Gemini's generation config
- When called, yields an EndCall event with a goodbye message
- Requires user confirmation before ending (per system prompt)

### Response Streaming

- Gemini responses are streamed token-by-token
- Each text chunk yields an `AgentResponse` event
- Full responses are logged for debugging
- Supports async streaming via `AsyncGenerator`

### Voice Optimization

The system prompt enforces voice-friendly responses:
- Limit to 1-2 sentences, less than 35 words
- No emojis or abbreviations
- Spell out units and dates
- One question at a time
- Concise but polite tone

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (required for LLM) | None |
| `MODEL_ID` | Gemini model to use | gemini-2.5-flash |
| `PORT` | Server port for local development | 8000 |

## Development Workflow

### Local Testing

```bash
# Set environment variables
export GEMINI_API_KEY=your_api_key_here

# Run with uv (recommended)
PORT=8000 uv run python main.py

# Test in another terminal
cartesia chat 8000
```

### Testing

- Test file: `test_basic_chat.py`
- Uses pytest with asyncio support
- Development dependencies include pytest-cov, pytest-xdist, pytest-repeat

### Dependencies

Core dependencies (from pyproject.toml):
- cartesia-line (the Cartesia Line SDK)
- google-genai>=1.26.0 (Gemini client)
- python-dotenv>=1.0.0 (environment variables)
- loguru>=0.7.0 (logging)
- aiohttp>=3.12.0 (async HTTP)
- uvicorn==0.35.0 (ASGI server)

## Code Patterns

### Node Pattern

The ChatNode follows the Cartesia Line **ReasoningNode** pattern:
- Extends `ReasoningNode` base class
- Implements `process_context()` to handle LLM streaming
- Uses `ConversationContext` for conversation state
- Yields `AgentResponse` and `EndCall` events

### Bridge Pattern

The Bridge connects events to node methods:
```python
conversation_bridge.on(UserTranscriptionReceived).map(conversation_node.add_event)

conversation_bridge.on(UserStoppedSpeaking)
    .interrupt_on(UserStartedSpeaking, handler=conversation_node.on_interrupt_generate)
    .stream(conversation_node.generate)
    .broadcast()
```

### Error Handling

- Missing Gemini API key: Provides canned responses instead of crashing
- Logging via loguru for debugging (user messages, agent responses, tool calls)
- Async error handling via try/except in async generators

## Deployment

Deployment is managed via Cartesia CLI:
```bash
cartesia auth login
cartesia deploy
```

Configuration is in `cartesia.toml` (single line file).

## Modification Guidelines

When modifying this codebase:

1. **Keep responses voice-friendly**: Always enforce brevity and clarity for spoken output
2. **Maintain streaming**: Don't break the async generator pattern in `process_context()`
3. **Log important events**: Use loguru to log user messages, agent responses, and tool calls
4. **Handle missing API keys gracefully**: Keep the canned response fallback
5. **Test locally first**: Use `cartesia chat 8000` to test before deploying
6. **Respect context limits**: Don't overflow the conversation history (max 100 turns)
7. **Follow tool call patterns**: When adding new tools, follow the EndCallTool pattern

## Common Tasks

### Changing the System Prompt
Edit `config.py` → `SYSTEM_PROMPT` variable

### Changing the LLM Model
Set `MODEL_ID` environment variable or modify `DEFAULT_MODEL_ID` in `config.py`

### Adding New Tools
1. Define tool schema (Gemini function format)
2. Add to `generation_config.tools` in ChatNode.__init__()
3. Handle tool calls in `process_context()` via `msg.function_calls`

### Adjusting Temperature
Modify `DEFAULT_TEMPERATURE` in `config.py` or pass to ChatNode constructor

### Changing Max Context Length
Pass `max_context_length` to ChatNode constructor in `main.py`

## Debug Tips

- Check logs for user message and agent response pairs
- Verify Gemini API key is set if seeing canned responses
- Use `logger.info()` statements to trace event flow
- Test locally with `cartesia chat 8000` before deploying
- Check conversation context length if responses seem confused

## Related Documentation

- [Cartesia Line Docs](https://docs.cartesia.ai/line/)
- [Google Gemini API Docs](https://ai.google.dev/gemini-api/docs)
- [Cartesia Voice Agents](https://cartesia.ai/)
