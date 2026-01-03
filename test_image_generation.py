"""
Tests for example_image_generation.py

Run this via:
```
uv sync --extra dev
uv run pytest test_image_generation.py
```
"""

import os
import tempfile
from unittest.mock import Mock, patch, mock_open

import pytest
from google.genai import types

from example_image_generation import generate_content_with_image


@pytest.fixture
def mock_client():
    """Create a mock Gemini client."""
    mock = Mock()
    mock.models.generate_content.return_value = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[types.Part(text="This is a test image showing a landscape.")],
                    role="model"
                ),
                finish_reason="STOP"
            )
        ]
    )
    return mock


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes (simple JPEG header)."""
    # Minimal JPEG structure for testing:
    # \xff\xd8 = SOI (Start of Image)
    # \xff\xe0 = APP0 marker (JFIF)
    # JFIF header data
    # \xff\xd9 = EOI (End of Image)
    return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9'


def test_generate_content_with_image_jpeg(mock_client, sample_image_bytes, tmp_path):
    """Test generating content with a JPEG image."""
    # Create a temporary image file
    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(sample_image_bytes)
    
    with patch('example_image_generation.genai.Client', return_value=mock_client):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            result = generate_content_with_image(str(image_file), "What's this image about?")
    
    # Verify the client was called correctly
    mock_client.models.generate_content.assert_called_once()
    
    # Get the call arguments
    call_args = mock_client.models.generate_content.call_args
    
    # Verify model parameter
    assert call_args.kwargs['model'] == 'gemini-2.0-flash'
    
    # Verify contents structure
    contents = call_args.kwargs['contents']
    assert len(contents) == 1
    assert isinstance(contents[0], types.Content)
    
    # Verify parts
    parts = contents[0].parts
    assert len(parts) == 2
    assert parts[0].text == "What's this image about?"
    assert parts[1].inline_data is not None
    assert parts[1].inline_data.mime_type == "image/jpeg"
    assert parts[1].inline_data.data == sample_image_bytes
    
    # Verify config is None
    assert call_args.kwargs['config'] is None
    
    # Verify result
    assert result.text == "This is a test image showing a landscape."


def test_generate_content_with_image_png(mock_client, tmp_path):
    """Test generating content with a PNG image."""
    # Create a temporary PNG file
    image_file = tmp_path / "test.png"
    png_bytes = b'\x89PNG\r\n\x1a\n'  # PNG header
    image_file.write_bytes(png_bytes)
    
    with patch('example_image_generation.genai.Client', return_value=mock_client):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            result = generate_content_with_image(str(image_file))
    
    # Verify mime type is correct for PNG
    call_args = mock_client.models.generate_content.call_args
    parts = call_args.kwargs['contents'][0].parts
    assert parts[1].inline_data.mime_type == "image/png"


def test_generate_content_missing_api_key(tmp_path):
    """Test that missing API key raises an error."""
    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(b'fake image data')
    
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable must be set"):
            generate_content_with_image(str(image_file))


def test_generate_content_file_not_found():
    """Test that non-existent file raises an error."""
    with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
        with pytest.raises(FileNotFoundError):
            generate_content_with_image("/nonexistent/file.jpg")


def test_custom_prompt(mock_client, sample_image_bytes, tmp_path):
    """Test using a custom prompt."""
    image_file = tmp_path / "test.jpg"
    image_file.write_bytes(sample_image_bytes)
    
    custom_prompt = "Describe this image in detail"
    
    with patch('example_image_generation.genai.Client', return_value=mock_client):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            generate_content_with_image(str(image_file), custom_prompt)
    
    # Verify custom prompt was used
    call_args = mock_client.models.generate_content.call_args
    parts = call_args.kwargs['contents'][0].parts
    assert parts[0].text == custom_prompt


def test_unsupported_image_format(tmp_path):
    """Test that unsupported image format raises an error."""
    image_file = tmp_path / "test.bmp"
    image_file.write_bytes(b'fake image data')
    
    with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
        with pytest.raises(ValueError, match="Unsupported image format"):
            generate_content_with_image(str(image_file))
