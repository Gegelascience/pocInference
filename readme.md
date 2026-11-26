endpoints openai compatible

POST /v1/chat/completions

curl https://your-custom-endpoint.com/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "How can I integrate OpenAI-compatible APIs?"}
    ]
  }'

GET /v1/models
curl https://your-custom-endpoint.com/v1/models \
  -H "Authorization: Bearer your-api-key"