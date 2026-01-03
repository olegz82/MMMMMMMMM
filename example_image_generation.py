"""
Example demonstrating how to generate content with both text and images using Google Gemini.

This is the Python equivalent of the following Go code:

```go
parts := []*genai.Part{
  {Text: "What's this image about?"},
  {InlineData: &genai.Blob{Data: imageBytes, MIMEType: "image/jpeg"}},
}
result, err := client.Models.GenerateContent(ctx, "gemini-2.0-flash", []*genai.Content{{Parts: parts}}, nil)
```

Usage:
    python example_image_generation.py path/to/image.jpg
"""

import os
import sys
from google import genai
from google.genai import types


def generate_content_with_image(image_path: str, prompt: str = "What's this image about?"):
    """
    Generate content using both text and image input.
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt to send along with the image
        
    Returns:
        The generated response from Gemini
    """
    # Initialize the Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable must be set")
    
    client = genai.Client(api_key=api_key)
    
    # Read the image file
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Determine MIME type based on file extension
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        mime_type = "image/jpeg"
    elif image_path.lower().endswith('.png'):
        mime_type = "image/png"
    elif image_path.lower().endswith('.gif'):
        mime_type = "image/gif"
    elif image_path.lower().endswith('.webp'):
        mime_type = "image/webp"
    else:
        raise ValueError(
            f"Unsupported image format. Supported formats: .jpg, .jpeg, .png, .gif, .webp. "
            f"Got: {os.path.splitext(image_path)[1]}"
        )
    
    # Create parts: text + image (equivalent to Go code)
    parts = [
        types.Part(text=prompt),
        types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime_type)),
    ]
    
    # Create content with the parts
    content = types.Content(parts=parts)
    
    # Generate content using gemini-2.0-flash model
    result = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[content],
        config=None,
    )
    
    return result


def main():
    """Main function to demonstrate the example."""
    if len(sys.argv) < 2:
        print("Usage: python example_image_generation.py <path_to_image>")
        print("Example: python example_image_generation.py photo.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)
    
    print(f"Analyzing image: {image_path}")
    print("Sending request to Gemini...")
    
    try:
        result = generate_content_with_image(image_path)
        
        print("\n=== Gemini Response ===")
        print(result.text)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
