#!/bin/bash
# Setup a project to use the gateway
# Usage: bash scripts/setup_project.sh <project-name> <llm-module-path>

set -e

PROJECT_NAME="${1:-my-project}"
LLM_MODULE="${2:-.env}"

echo "Setting up $PROJECT_NAME to use the gateway..."

# Create or update .env file
ENV_FILE="$LLM_MODULE"
if [ "$ENV_FILE" = ".env" ]; then
    if [ ! -f "$ENV_FILE" ]; then
        touch "$ENV_FILE"
    fi

    # Remove existing gateway settings
    sed -i.bak '/^OPENAI_BASE_URL=/d; /^OPENAI_API_KEY=wd-/d' "$ENV_FILE" 2>/dev/null || \
    sed -i '' '/^OPENAI_BASE_URL=/d; /^OPENAI_API_KEY=wd-/d' "$ENV_FILE"

    # Add new settings
    cat >> "$ENV_FILE" << GATEWAY_CONFIG

# Gateway configuration (auto-generated)
OPENAI_BASE_URL=https://llmcostwatchdog.onrender.com/v1
OPENAI_API_KEY=wd-${PROJECT_NAME}
GATEWAY_CONFIG

    echo "✓ Updated $ENV_FILE"
fi

echo ""
echo "Configuration:"
echo "  Project name: $PROJECT_NAME"
echo "  Base URL: https://llmcostwatchdog.onrender.com/v1"
echo "  API Key: wd-$PROJECT_NAME"
echo ""
echo "Usage in code:"
echo "  from openai import OpenAI"
echo "  client = OpenAI()"
echo "  response = client.chat.completions.create("
echo "      model='group:ladder',"
echo "      messages=[{'role': 'user', 'content': '...'}]"
echo "  )"
echo ""
echo "View live calls:"
echo "  https://llmcostwatchdog.onrender.com/calls?source=live"
echo ""
echo "Ready to go! Run your project normally."
