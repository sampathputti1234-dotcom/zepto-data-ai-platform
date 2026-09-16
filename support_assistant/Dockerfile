FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY support_assistant/support_assistant.py .
COPY support_assistant/docs ./docs

ENV MOCK_LLM=1

EXPOSE 7860

CMD ["uvicorn", "support_assistant:api", "--host", "0.0.0.0", "--port", "7860"]