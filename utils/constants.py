DEFAULT_MEMORY_CONFIG = {
    "type": "sqlite",
    "path": "/data/cache/memory.db",
}

VISION_SYSTEM_MESSAGE = (
    "[IMPORTANT — IMAGE PROCESSING]: The message contains inline base64 image data (image_url). "
    "You MUST analyze the image directly from the provided base64 data to answer the user's question. "
    "Do NOT call any file reading tool to re-read an image that is already "
    "embedded inline in the message — the full image content is already available.\n"
    "Image filepaths (e.g. '/attachments/uuid/filename.png') are internal storage identifiers, NOT URLs. "
    "NEVER construct URLs or markdown image links like ![](url) from filepaths — any such link will be broken. "
    "When asked to show or list existing images, refer to them by filepath as plain text. "
    "Do NOT call generate_image or edit_image to 'show' existing images — only call these tools to CREATE new images."
)

# Template for attachment system message - conversation_id is filled in at runtime
ATTACHMENT_SYSTEM_MESSAGE_TEMPLATE = (
    "[ATTACHMENTS]: All conversation file attachments are stored in bucket 'attachments' "
    "under folder '{conversation_id}'. "
    "This storage is READ-ONLY — saving, uploading, or writing any files into the 'attachments' bucket "
    "is strictly PROHIBITED. Never use bucket 'attachments' as a destination for any tool calls that "
    "create or store files. "
    "Use the Attachments toolkit ONLY when the user explicitly refers to a file they uploaded or shared "
    "in this conversation. For all other file requests, search across all available artifact toolkits."
)

