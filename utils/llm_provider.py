"""
LLM Provider Abstraction Layer

Supports multiple LLM providers (Ollama, Gemini) with a unified interface.
Provider selection is controlled via config.yaml.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json


class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider (local models)"""
    
    def __init__(self, config: dict):
        """
        Initialize Ollama provider.
        
        Args:
            config: Ollama configuration from config.yaml
        """
        try:
            import ollama
            self.client = ollama.Client(host=config.get('base_url', 'http://localhost:11434'))
            self.model = config.get('model', 'qwen2.5:7b')
            print(f"INFO: Initialized Ollama provider with model: {self.model}")
        except ImportError:
            raise ImportError(
                "Ollama package not installed. Install with: pip install ollama"
            )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate chat completion using Ollama"""
        try:
            options = {
                'temperature': temperature,
                'num_predict': max_tokens,
            }
            
            if stop:
                options['stop'] = stop
            
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options
            )
            
            return response['message']['content']
            
        except Exception as e:
            print(f"ERROR: Ollama API call failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Ollama models.
        Uses simple word-based estimation (roughly 1.3 tokens per word).
        """
        words = len(text.split())
        return int(words * 1.3)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""
    
    def __init__(self, config: dict):
        """
        Initialize OpenAI provider.
        
        Args:
            config: OpenAI configuration from config.yaml
        """
        try:
            from openai import OpenAI
            import tiktoken
            
            api_key = config.get('api_key')
            if not api_key or api_key == 'YOUR_OPENAI_API_KEY_HERE':
                raise ValueError(
                    "OpenAI API key not configured. Add your key to config.yaml"
                )
            
            self.client = OpenAI(api_key=api_key)
            self.model = config.get('model', 'gpt-3.5-turbo')
            self.encoding = tiktoken.encoding_for_model(self.model)
            print(f"INFO: Initialized OpenAI provider with model: {self.model}")
            
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai tiktoken"
            )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate chat completion using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"ERROR: OpenAI API call failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's tokenizer"""
        return len(self.encoding.encode(text))


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider (using new google-genai SDK)"""
    
    def __init__(self, config: dict):
        """
        Initialize Gemini provider.
        
        Args:
            config: Gemini configuration from config.yaml
        """
        try:
            from google import genai
            from google.genai import types
            
            api_key = config.get('api_key')
            if not api_key or api_key == 'YOUR_GEMINI_API_KEY_HERE':
                raise ValueError(
                    "Gemini API key not configured. Add your key to config.yaml"
                )
            
            self.client = genai.Client(api_key=api_key)
            self.model_name = config.get('model', 'gemini-2.5-flash')  # Use 2.5-flash with thinking disabled
            print(f"INFO: Initialized Gemini provider with model: {self.model_name}")
            print(f"INFO: Thinking mode disabled (thinking_budget=0)")
            
        except ImportError:
            raise ImportError(
                "Gemini package not installed. Install with: pip install google-genai"
            )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate chat completion using Gemini"""
        try:
            from google.genai import types
            
            # Convert messages to Gemini format
            contents = []
            system_instruction = None
            
            for msg in messages:
                role = msg['role']
                content = msg['content']
                
                if role == 'system':
                    system_instruction = content
                elif role == 'user':
                    contents.append(types.Content(
                        role='user',
                        parts=[types.Part(text=content)]
                    ))
                elif role == 'assistant':
                    contents.append(types.Content(
                        role='model',
                        parts=[types.Part(text=content)]
                    ))
            
            # Build generation config with thinking DISABLED
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction if system_instruction else None,
                stop_sequences=stop if stop else None,
                # DISABLE THINKING MODE: Set thinking_budget=0 for no thinking
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            # Gemini response has a direct .text attribute
            if hasattr(response, 'text') and response.text:
                return response.text
            
            # If text is None, might have hit MAX_TOKENS - check finish reason
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
                if 'MAX_TOKENS' in str(finish_reason):
                    print(f"WARNING: Gemini hit MAX_TOKENS limit ({max_tokens}). Consider increasing max_tokens in config.yaml")
            
            print(f"ERROR: Gemini returned no text")
            return None
            
        except Exception as e:
            print(f"ERROR: Gemini API call failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Gemini models.
        Uses simple word-based estimation (roughly 1.3 tokens per word).
        """
        words = len(text.split())
        return int(words * 1.3)


def get_llm_provider(config: dict) -> LLMProvider:
    """
    Factory function to create LLM provider based on config.
    
    Args:
        config: Full configuration dict from config.yaml
        
    Returns:
        Initialized LLM provider instance
    """
    provider_type = config['llm']['provider'].lower()
    
    if provider_type == 'ollama':
        return OllamaProvider(config['llm']['ollama'])
    elif provider_type == 'openai':
        return OpenAIProvider(config['llm']['openai'])
    elif provider_type == 'gemini':
        return GeminiProvider(config['llm']['gemini'])
    else:
        raise ValueError(
            f"Unknown provider: {provider_type}. Supported: 'ollama', 'openai', 'gemini'"
        )
