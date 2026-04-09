# Based on https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb (MIT license)
import os
import sys
from typing import List, Optional, Dict, Any, Union

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.outputs import LLMResult

from pylon.core.tools import log

from .constants import VISION_SYSTEM_MESSAGE, ATTACHMENT_SYSTEM_MESSAGE_TEMPLATE


# Development mode flag - set via environment variable or config
_DEV_MODE_RELOAD = None


def is_dev_reload_enabled() -> bool:
    """Check if development mode SDK reload is enabled."""
    global _DEV_MODE_RELOAD
    if _DEV_MODE_RELOAD is None:
        _DEV_MODE_RELOAD = os.environ.get("ELITEA_SDK_DEV_RELOAD", "").lower() in ("1", "true", "yes")
    return _DEV_MODE_RELOAD


def set_dev_reload_enabled(enabled: bool) -> None:
    """Enable or disable development mode SDK reload."""
    global _DEV_MODE_RELOAD
    _DEV_MODE_RELOAD = enabled


def clear_sdk_modules(target_module: str = None) -> int:
    """
    Remove elitea_sdk modules from sys.modules cache to force fresh imports.

    This is useful during development to pick up SDK code changes without
    restarting the worker process.

    Args:
        target_module: If specified, only clear this module and its submodules.
                      If None, clears all elitea_sdk modules (may cause circular import issues).

    Returns:
        Number of modules cleared from cache.

    Usage:
        # Clear specific module tree (safer)
        clear_sdk_modules('elitea_sdk.runtime.clients')
        from elitea_sdk.runtime.clients.client import EliteAClient

        # Clear all (may cause issues with complex dependencies)
        clear_sdk_modules()
    """
    if not is_dev_reload_enabled():
        return 0

    if target_module:
        # Clear only the target module and its submodules
        sdk_modules = [
            key for key in list(sys.modules.keys())
            if key == target_module or key.startswith(target_module + '.')
        ]
    else:
        # Clear all SDK modules
        sdk_modules = [key for key in list(sys.modules.keys()) if key.startswith('elitea_sdk')]

    count = len(sdk_modules)

    for mod_name in sdk_modules:
        del sys.modules[mod_name]

    if count > 0:
        log.debug(f"[DEV_RELOAD] Cleared {count} elitea_sdk modules from cache")

    return count


def dev_reload_sdk(target_module: str = None):
    """
    Convenience function to clear SDK modules if dev mode is enabled.

    Call this before importing SDK modules in method functions to enable
    live code reloading during development.

    Args:
        target_module: Optional specific module path to reload (e.g., 'elitea_sdk.runtime.clients').
                      If None, clears all SDK modules.

    Example:
        def my_method(self):
            # Safer - only reload client module tree
            dev_reload_sdk('elitea_sdk.runtime.clients')
            from elitea_sdk.runtime.clients.client import EliteAClient

            # Or reload all (may have circular import issues)
            dev_reload_sdk()
    """
    clear_sdk_modules(target_module)


def extract_token_usage(response: LLMResult) -> Optional[Dict[str, int]]:
    """Extract token usage from LLMResult trying multiple possible locations.
    
    Returns dict with 'prompt_tokens' and 'completion_tokens' if found, None otherwise.
    Wrapped in try/except to prevent any exceptions from causing regressions.
    """
    try:
        # Method 1: Check llm_output (traditional location old models)
        if response.llm_output and 'token_usage' in response.llm_output:
            token_usage = response.llm_output['token_usage']
            return {
                'prompt_tokens': token_usage.get('prompt_tokens', token_usage.get('input_tokens', 0)),
                'completion_tokens': token_usage.get('completion_tokens', token_usage.get('output_tokens', 0))
            }
        
        # Method 2: Check message.usage_metadata (modern models)
        if response.generations:
            for generation_list in response.generations:
                for gen in generation_list:
                    if hasattr(gen, 'message') and hasattr(gen.message, 'usage_metadata'):
                        usage_meta = gen.message.usage_metadata
                        if usage_meta:
                            return {
                                'prompt_tokens': usage_meta.get('input_tokens', usage_meta.get('prompt_tokens', 0)),
                                'completion_tokens': usage_meta.get('output_tokens', usage_meta.get('completion_tokens', 0))
                            }
        
        return None
    except Exception as e:
        log.debug(f"Failed to extract token usage from API response: {e}")
        return None


# Mapping of provider-specific stop reasons to normalized 'length' value
LENGTH_STOP_REASONS = {
    'length',
    'max_tokens',
    'STOP_REASON_MAX_TOKENS',
}


def extract_finish_reason(response: LLMResult, generation_chunk: Dict = None) -> Optional[str]:
    """Extract finish/stop reason from LLMResult trying multiple possible locations.

    Different LLM providers return stop reasons in different formats:
    - OpenAI: generation_info.finish_reason = 'stop' | 'length' | 'tool_calls'
    - Anthropic: message.response_metadata.stop_reason = 'end_turn' | 'max_tokens' | 'tool_use'
    - Gemini: message.response_metadata.finish_reason = 'STOP' | 'STOP_REASON_MAX_TOKENS'

    Args:
        response: LLMResult object from LangChain callback
        generation_chunk: Optional pre-extracted generation chunk dict

    Returns:
        Normalized finish reason string or None if not found.
        Returns 'length' for any max token limit stop reason across providers.
    """
    try:
        finish_reason = None

        # Method 1: Check generation_chunk if provided (from thinking_steps)
        if generation_chunk:
            # OpenAI format: generation_info.finish_reason
            generation_info = generation_chunk.get('generation_info')
            if isinstance(generation_info, dict):
                finish_reason = generation_info.get('finish_reason')

            # Anthropic/modern format: message.response_metadata.stop_reason
            if not finish_reason:
                message = generation_chunk.get('message', {})
                if isinstance(message, dict):
                    response_metadata = message.get('response_metadata', {})
                    if isinstance(response_metadata, dict):
                        # Try stop_reason (Anthropic) then finish_reason (others)
                        finish_reason = response_metadata.get('stop_reason') or response_metadata.get('finish_reason')

        # Method 2: Check response.generations directly
        if not finish_reason and response and response.generations:
            for generation_list in response.generations:
                for gen in generation_list:
                    # Check generation_info (OpenAI style)
                    if hasattr(gen, 'generation_info') and gen.generation_info:
                        finish_reason = gen.generation_info.get('finish_reason')
                        if finish_reason:
                            break

                    # Check message.response_metadata (Anthropic/modern style)
                    if not finish_reason and hasattr(gen, 'message'):
                        msg = gen.message
                        if hasattr(msg, 'response_metadata') and msg.response_metadata:
                            resp_meta = msg.response_metadata
                            if isinstance(resp_meta, dict):
                                finish_reason = resp_meta.get('stop_reason') or resp_meta.get('finish_reason')
                                if finish_reason:
                                    break
                if finish_reason:
                    break

        # Method 3: Check llm_output (legacy format)
        if not finish_reason and response and response.llm_output:
            finish_reason = response.llm_output.get('stop_reason') or response.llm_output.get('finish_reason')

        # Normalize length-related stop reasons
        if finish_reason and finish_reason in LENGTH_STOP_REASONS:
            return 'length'

        return finish_reason

    except Exception as e:
        log.debug(f"Failed to extract finish reason from API response: {e}")
        return None


def num_tokens_from_messages(messages: List[BaseMessage] | List[str], model="gpt-3.5-turbo-0613", is_chunk=False):
    """Return the number of tokens used by a list of messages.
    
    Args:
        messages: List of BaseMessage objects or strings
        model: Model name for encoding
        is_chunk: If True, only count content tokens without message overhead (for streaming chunks)
    """
    import tiktoken
    #
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        log.debug("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    #
    if not is_chunk:
        if model in {
            "gpt-3.5-turbo-0125",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-08-06",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif "gpt-3.5-turbo" in model:
            log.debug("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
            return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125", is_chunk=is_chunk)
        elif "gpt-4o-mini" in model:
            log.debug("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
            return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18", is_chunk=is_chunk)
        elif "gpt-4o" in model:
            log.debug("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
            return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06", is_chunk=is_chunk)
        elif "gpt-4" in model:
            log.debug("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return num_tokens_from_messages(messages, model="gpt-4-0613", is_chunk=is_chunk)
        else:
            log.debug("num_tokens_from_messages() is not implemented for model %s.", model)
            tokens_per_message = 3
            tokens_per_name = 1
            # ^ dummy values
    else:
        # For chunks, don't add overhead
        tokens_per_message = 0
        tokens_per_name = 0
    #
    num_tokens = 0
    for message_item in messages:
        num_tokens += tokens_per_message

        if isinstance(message_item, str):
            num_tokens += len(encoding.encode(str(message_item)))
        else:
            # Handle ChatGenerationChunk and similar wrapper objects
            if hasattr(message_item, 'message'):
                message_item = message_item.message
            
            # For BaseMessage objects, only count fields that are sent to the API
            # Don't use .dict() iteration as it includes metadata fields not sent to OpenAI
            message = message_item.dict()
            
            if not is_chunk:
                # Count role field (from message type)
                role = message.get('type', 'user')
                num_tokens += len(encoding.encode(role))
            
            # Handle content field
            content = message.get('content', message.get('text', ''))
            if isinstance(content, list):
                # Extract text from content blocks like [{'type': 'text', 'text': 'hi'}]
                # Image blocks like {'type': 'image_url', 'image_url': {...}} are ignored for token counting
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        text_parts.append(str(block['text']))
                content_text = ''.join(text_parts)
            else:
                content_text = str(content)
            
            num_tokens += len(encoding.encode(content_text))
            
            if not is_chunk:
                # Count the name field separately if it exists
                if message.get('name'):
                    num_tokens += len(encoding.encode(str(message['name'])))
                    num_tokens += tokens_per_name

    if not is_chunk:
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def prepend_vision_system_message(
    chat_history: List[Union[Dict[str, Any], Any]]
) -> List[Union[Dict[str, Any], Any]]:
    """
    Prepend VISION_SYSTEM_MESSAGE to chat_history, ensuring only one system message.
    
    Anthropic models require the system message to be FIRST and don't support
    multiple consecutive system messages. This function:
    - If chat_history starts with a system message (dict or LangChain), merges vision content into it
    - Otherwise, prepends a new system message
    
    Args:
        chat_history: List of messages (dicts with 'role'/'content' or LangChain message objects)
        
    Returns:
        Modified chat_history with vision system message prepended/merged
    """
    if not chat_history:
        return [{'role': 'system', 'content': VISION_SYSTEM_MESSAGE}]
    
    first_msg = chat_history[0]
    
    # Check if first message is a system message (dict format)
    if isinstance(first_msg, dict) and first_msg.get('role') == 'system':
        first_msg['content'] = VISION_SYSTEM_MESSAGE + '\n\n' + first_msg.get('content', '')
        return chat_history
    
    # Check if first message is a LangChain SystemMessage
    if isinstance(first_msg, SystemMessage):
        first_msg.content = VISION_SYSTEM_MESSAGE + '\n\n' + first_msg.content
        return chat_history
    
    # No existing system message - prepend new one (dict format for compatibility)
    return [{'role': 'system', 'content': VISION_SYSTEM_MESSAGE}] + chat_history


def prepend_attachment_system_message(
    chat_history: List[Union[Dict[str, Any], Any]],
    conversation_id: str
) -> List[Union[Dict[str, Any], Any]]:
    """
    Prepend attachment system message to chat_history describing file storage location.
    
    This helps the LLM understand where conversation attachments are stored and how
    to use the listFiles tool with the correct folder parameter.
    
    Anthropic models require the system message to be FIRST and don't support
    multiple consecutive system messages. This function:
    - If chat_history starts with a system message (dict or LangChain), merges attachment content into it
    - Otherwise, prepends a new system message
    
    Args:
        chat_history: List of messages (dicts with 'role'/'content' or LangChain message objects)
        conversation_id: The conversation ID to use as folder path for attachments
        
    Returns:
        Modified chat_history with attachment system message prepended/merged
    """
    attachment_msg = ATTACHMENT_SYSTEM_MESSAGE_TEMPLATE.format(conversation_id=conversation_id)
    
    if not chat_history:
        return [{'role': 'system', 'content': attachment_msg}]
    
    first_msg = chat_history[0]
    
    # Check if first message is a system message (dict format)
    if isinstance(first_msg, dict) and first_msg.get('role') == 'system':
        first_msg['content'] = attachment_msg + '\n\n' + first_msg.get('content', '')
        return chat_history
    
    # Check if first message is a LangChain SystemMessage
    if isinstance(first_msg, SystemMessage):
        first_msg.content = attachment_msg + '\n\n' + first_msg.content
        return chat_history
    
    # No existing system message - prepend new one (dict format for compatibility)
    return [{'role': 'system', 'content': attachment_msg}] + chat_history

def normalize_mcp_toolkit_name(name: str) -> str:
    """
    Normalize MCP toolkit name/type for case-insensitive matching.

    Converts to lowercase, removes/normalizes whitespace, and removes the 'mcp_' prefix
    to handle variations in toolkit naming (e.g., "mcp_epam_presales" -> "epam_presales",
    "Epam Presales" -> "epam_presales").

    Args:
        name: The toolkit name or type to normalize

    Returns:
        Normalized name (lowercase, spaces converted to underscores, 'mcp_' prefix removed)
    """
    if not name:
        return ""
    # Convert to lowercase, replace spaces with underscores, strip
    normalized = name.lower().replace(" ", "_").strip()
    # Remove 'mcp_' prefix if present
    if normalized.startswith("mcp_"):
        normalized = normalized[4:]
    return normalized


def get_mcp_server_settings(toolkit_name: str) -> Optional[Dict[str, Any]]:
    """
    Get all configuration settings for a remote MCP server from pylon configuration.

    Reads the `mcp_servers` section from pylon.yml configuration and returns
    all settings for the specified toolkit using normalized case-insensitive matching.
    This includes credentials (client_id, client_secret) and any other configuration
    parameters defined for the MCP server.

    Example pylon.yml:
        mcp_servers:
          Epam Presales:
            client_id: "c1e199f6-ae8d-40a3-9c0c-b921e1af6474"
            client_secret: "nnSGxaeKHF1GvaL3"
            timeout: 30
            base_url: "https://api.example.com"
            custom_param: "value"

    Args:
        toolkit_name: The name of the MCP toolkit (e.g., "Epam Presales")

    Returns:
        Dict with all configuration settings if found, None otherwise.
        Always includes at least 'client_id' and 'client_secret' if the server is found.
    """
    try:
        log.info(f"Getting MCP server configuration for '{toolkit_name}'")
        from tools import this  # pylint: disable=E0401,C0415

        # Access pylon configuration through module descriptor
        module = this.module  # Get current indexer_worker module
        if not hasattr(module, 'descriptor') or not hasattr(module.descriptor, 'config'):
            log.debug("Cannot access pylon configuration - descriptor not available")
            return None

        config = module.descriptor.config
        mcp_servers = config.get('mcp_servers', {})
        log.info(f"Found servers:\n{mcp_servers}")
        log.debug(f"Found pylon configuration with {len(mcp_servers)} mcp_servers sections")

        if not mcp_servers:
            log.debug("No mcp_servers configuration found in pylon.yml")
            return None

        # Normalize the search name
        normalized_search = normalize_mcp_toolkit_name(toolkit_name)

        # Search for matching server config with normalized matching
        for server_name, server_config in mcp_servers.items():
            log.info(f"Checking MCP server '{server_name}' against search '{toolkit_name}' (normalized: '{normalized_search}')")
            if normalize_mcp_toolkit_name(server_name) == normalized_search:
                if not server_config or not isinstance(server_config, dict):
                    log.debug(f"MCP server '{server_name}' found but has invalid configuration")
                    return None

                log.debug(f"Found MCP configuration for '{toolkit_name}' in pylon config with {len(server_config)} settings")
                return dict(server_config)

        log.debug(f"No MCP server configuration found for '{toolkit_name}' in pylon config")
        return None

    except Exception as e:
        log.warning(f"Failed to get MCP server configuration from pylon config: {e}")
        return None


def resolve_mcp_credentials(toolkit_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve MCP toolkit configuration with fallback to pylon configuration.

    This function processes:
    1. Pre-built MCP toolkits (where type starts with 'mcp_') - uses toolkit_type for lookup
    2. Generic 'mcp' toolkits with server_name in settings - uses server_name for lookup

    Priority order for each setting:
    1. If toolkit settings already have a value - use it (no override)
    2. Otherwise, try to get from pylon configuration's mcp_servers section
    3. If not found, return toolkit_config unchanged

    This allows centralized configuration management in pylon.yml for remote MCP
    toolkits (including credentials, timeouts, URLs, etc.) while still supporting
    per-toolkit customization in settings.

    Args:
        toolkit_config: Toolkit configuration dict with 'type', 'toolkit_name', 'settings', etc.
            For pre-built MCP: {'type': 'mcp_epam_presales', 'settings': {...}}
            For generic MCP with server_name: {'type': 'mcp', 'settings': {'server_name': 'pre_built_toolkit', ...}}

    Returns:
        Updated toolkit_config with settings injected from pylon config where missing
    """
    try:
        toolkit_type = toolkit_config.get('type', '')
        settings = toolkit_config.get('settings', {})
        server_name = settings.get('server_name') if settings else None

        log.info(f"Resolving MCP configuration for toolkit type '{toolkit_type}', server_name='{server_name}'")

        # Determine what to use for config lookup:
        # 1. If server_name is provided in settings, use it (for pre-built MCP via generic 'mcp' type)
        # 2. Otherwise, if toolkit_type starts with 'mcp_', use toolkit_type
        # 3. If neither, skip processing
        if server_name:
            config_lookup_key = server_name
            log.debug(f"Using server_name '{server_name}' for MCP config lookup")
        elif toolkit_type and toolkit_type.startswith('mcp_'):
            config_lookup_key = toolkit_type
            log.debug(f"Using toolkit_type '{toolkit_type}' for MCP config lookup")
        else:
            log.debug(f"Skipping MCP config resolution - toolkit type '{toolkit_type}' is not a pre-built MCP toolkit and no server_name provided")
            return toolkit_config

        # Look up configuration using the determined key
        pylon_config = get_mcp_server_settings(config_lookup_key)
        log.info(f"Resolving MCP configuration for '{config_lookup_key}' - found pylon config: {bool(pylon_config)}")
        if not pylon_config:
            # No pylon config found, return unchanged
            return toolkit_config

        # Merge settings: pylon config values as defaults, toolkit settings take priority
        # Start with pylon config as base, then overlay existing toolkit settings
        merged_settings = dict(pylon_config)  # Copy pylon config as defaults

        # Overlay existing toolkit settings (they take priority over pylon config)
        if settings:
            for key, value in settings.items():
                if value:  # Only override if value is truthy
                    merged_settings[key] = value

        # Track what was injected from pylon config
        injected_settings = [key for key in pylon_config.keys() if key not in settings or not settings.get(key)]

        if injected_settings:
            log.debug(f"Injected {len(injected_settings)} setting(s) for MCP config '{config_lookup_key}' from pylon config: {', '.join(injected_settings)}")
        else:
            log.debug(f"MCP config '{config_lookup_key}' already has all settings, no injection needed")

        # Explicitly set the merged settings on toolkit_config
        toolkit_config['settings'] = merged_settings

        return toolkit_config

    except Exception as e:
        log.warning(f"Failed to resolve MCP configuration: {e}")
        return toolkit_config

def mask_secret(secret: str, visible_chars: int = 4) -> str:
    """
    Mask a secret string, showing only the last N characters.

    Args:
        secret: The secret string to mask
        visible_chars: Number of characters to show at the end (default: 4)

    Returns:
        Masked string like '****abcd' or fully masked if shorter than visible_chars
    """
    if not secret:
        return ""
    if len(secret) >= visible_chars:
        return '*' * (len(secret) - visible_chars) + secret[-visible_chars:]
    return '*' * len(secret)