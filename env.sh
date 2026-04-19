#!/bin/bash
# Environment configuration for MOPS LLM Task Agent with Foundry/Azure AI

# Required: Foundry API endpoint
# Example: https://your-resource.services.ai.azure.com
export FOUNDRY_BASE_URL="NEED REPLACE"

# Required: Foundry API key
export FOUNDRY_API_KEY="NEED REPLACE"

# Required: Model/deployment name
# Example: gpt-4, gpt-35-turbo, your-deployment-name
export FOUNDRY_MODEL="gpt-4o"

# Optional: API version (defaults to 2024-05-01-preview if not set)
export FOUNDRY_API_VERSION="2024-12-01-preview"

# Alternative Azure variable names (also supported):
# export AZURE_INFERENCE_ENDPOINT="YOUR_ENDPOINT"
# export AZURE_INFERENCE_API_KEY="YOUR_KEY"
# export AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT"
# export AZURE_OPENAI_API_KEY="YOUR_KEY"
# export AZURE_OPENAI_DEPLOYMENT="YOUR_DEPLOYMENT"

echo "Foundry environment variables set:"
echo "  FOUNDRY_BASE_URL: ${FOUNDRY_BASE_URL}"
echo "  FOUNDRY_MODEL: ${FOUNDRY_MODEL}"
echo "  FOUNDRY_API_VERSION: ${FOUNDRY_API_VERSION}"
echo "  FOUNDRY_API_KEY: [set]"
