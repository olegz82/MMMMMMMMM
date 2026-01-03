# Image Content Generation Example

This example demonstrates how to use Google Gemini with multimodal input (text + images) in Python.

## Overview

The `example_image_generation.py` file shows the Python equivalent of this Go code:

```go
parts := []*genai.Part{
  {Text: "What's this image about?"},
  {InlineData: &genai.Blob{Data: imageBytes, MIMEType: "image/jpeg"}},
}
result, err := client.Models.GenerateContent(ctx, "gemini-2.0-flash", []*genai.Content{{Parts: parts}}, nil)
```

## Python Implementation

In Python, the equivalent code looks like this:

```python
from google import genai
from google.genai import types

# Initialize client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Read image
with open(image_path, "rb") as f:
    image_bytes = f.read()

# Create parts: text + image
parts = [
    types.Part(text="What's this image about?"),
    types.Part(inline_data=types.Blob(data=image_bytes, mime_type="image/jpeg")),
]

# Create content
content = types.Content(parts=parts)

# Generate content
result = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[content],
    config=None,
)

print(result.text)
```

## Usage

1. Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```

2. Run the example with an image:
   ```bash
   python example_image_generation.py path/to/your/image.jpg
   ```

3. Or use it in your own code:
   ```python
   from example_image_generation import generate_content_with_image
   
   result = generate_content_with_image("photo.jpg", "Describe this image")
   print(result.text)
   ```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

## Testing

Run the tests to verify the implementation:

```bash
uv sync --extra dev
uv run pytest test_image_generation.py -v
```

Or with pip:

```bash
pip install pytest
pytest test_image_generation.py -v
```

## API Mapping: Go to Python

| Go Syntax | Python Syntax |
|-----------|---------------|
| `genai.Part{Text: "..."}` | `types.Part(text="...")` |
| `genai.Blob{Data: bytes, MIMEType: "..."}` | `types.Blob(data=bytes, mime_type="...")` |
| `genai.Part{InlineData: &blob}` | `types.Part(inline_data=blob)` |
| `genai.Content{Parts: parts}` | `types.Content(parts=parts)` |
| `client.Models.GenerateContent(ctx, model, contents, config)` | `client.models.generate_content(model=model, contents=contents, config=config)` |

## Key Differences

1. **Naming conventions**: 
   - Go uses PascalCase for struct fields (e.g., `MIMEType`, `InlineData`)
   - Python uses snake_case for parameters (e.g., `mime_type`, `inline_data`)

2. **Pointers**: 
   - Go uses pointers (`&genai.Blob{...}`)
   - Python passes objects directly (`types.Blob(...)`)

3. **Context**: 
   - Go requires a context parameter (`ctx`)
   - Python handles context internally (or uses async/await for async operations)

4. **Error handling**:
   - Go returns `(result, err)`
   - Python raises exceptions

## See Also

- [Google Gemini Python SDK Documentation](https://ai.google.dev/gemini-api/docs)
- [Main chat application](./README.md)
