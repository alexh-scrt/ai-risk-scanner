"""Sample Python application demonstrating multiple AI SDK integrations.

This fixture file is used by integration tests to verify that the AI Risk
Inventory Scanner correctly detects a wide variety of AI provider references
including imports, SDK calls, model name references, and environment variable
usage patterns across multiple providers in a single file.
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# OpenAI SDK usage
# ---------------------------------------------------------------------------
import openai
from openai import OpenAI, AsyncOpenAI

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat_with_gpt(prompt: str) -> str:
    """Send a prompt to GPT-4o and return the response text.

    Args:
        prompt: The user prompt to send.

    Returns:
        The model's response as a string.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


def generate_embedding(text: str) -> list[float]:
    """Generate a text embedding using OpenAI's embedding model.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    result = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return result.data[0].embedding


def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe an audio file using OpenAI Whisper.

    Args:
        audio_file_path: Path to the audio file to transcribe.

    Returns:
        The transcription text.
    """
    with open(audio_file_path, "rb") as audio_file:
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return transcription.text


# ---------------------------------------------------------------------------
# Anthropic SDK usage
# ---------------------------------------------------------------------------
import anthropic
from anthropic import Anthropic, AsyncAnthropic

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def chat_with_claude(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Send a prompt to Claude and return the response.

    Args:
        prompt: The user prompt.
        system: The system prompt for Claude.

    Returns:
        Claude's response text.
    """
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def chat_with_claude_opus(prompt: str) -> str:
    """Send a prompt to Claude Opus for complex reasoning tasks.

    Args:
        prompt: The user prompt.

    Returns:
        Claude Opus response text.
    """
    message = anthropic_client.messages.create(
        model="claude-3-opus",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Cohere SDK usage
# ---------------------------------------------------------------------------
import cohere

co_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))


def generate_with_cohere(prompt: str) -> str:
    """Generate text using Cohere's Command model.

    Args:
        prompt: The input prompt.

    Returns:
        Generated text from Cohere.
    """
    response = co_client.chat(
        model="command-r-plus",
        message=prompt,
    )
    return response.text


def rerank_documents(query: str, documents: list[str]) -> list[dict[str, Any]]:
    """Re-rank documents using Cohere's rerank model.

    Args:
        query: The search query.
        documents: List of document strings to rerank.

    Returns:
        Reranked documents with relevance scores.
    """
    results = co_client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=5,
    )
    return [{"index": r.index, "score": r.relevance_score} for r in results.results]


# ---------------------------------------------------------------------------
# Hugging Face transformers usage
# ---------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import InferenceClient

hf_token = os.getenv("HF_TOKEN")


def load_local_model(model_name: str = "meta-llama/Llama-2-7b-hf") -> Any:
    """Load a local model from Hugging Face Hub.

    Args:
        model_name: The Hugging Face model identifier.

    Returns:
        The loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    return model, tokenizer


def run_hf_pipeline(text: str) -> str:
    """Run a text generation pipeline using Hugging Face.

    Args:
        text: The input text prompt.

    Returns:
        Generated text output.
    """
    generator = pipeline("text-generation", model="gpt2")
    output = generator(text, max_new_tokens=100)
    return output[0]["generated_text"]


def hf_inference_api(prompt: str, model: str = "mistralai/Mistral-7B-Instruct-v0.1") -> str:
    """Call the Hugging Face Inference API.

    Args:
        prompt: The input prompt.
        model: The model ID on Hugging Face Hub.

    Returns:
        Model response text.
    """
    client = InferenceClient(
        model=model,
        token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
    )
    return client.text_generation(prompt)


# ---------------------------------------------------------------------------
# LangChain orchestration
# ---------------------------------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain, ConversationChain
from langchain.agents import AgentExecutor, create_react_agent

langchain_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

langchain_claude = ChatAnthropic(
    model="claude-3-5-sonnet",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def langchain_chat(user_message: str) -> str:
    """Use LangChain to send a message and get a response.

    Args:
        user_message: The user's message.

    Returns:
        The model response string.
    """
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=user_message),
    ]
    response = langchain_llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Mistral AI usage
# ---------------------------------------------------------------------------
from mistralai import Mistral

mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))


def chat_with_mistral(prompt: str) -> str:
    """Send a prompt to Mistral Large and return the response.

    Args:
        prompt: The user prompt.

    Returns:
        Mistral's response text.
    """
    response = mistral_client.chat.complete(
        model="mistral-large",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def mistral_embed(text: str) -> list[float]:
    """Create an embedding using Mistral's embedding model.

    Args:
        text: The input text.

    Returns:
        Embedding vector.
    """
    response = mistral_client.embeddings.create(
        model="mistral-embed",
        inputs=[text],
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# AWS Bedrock usage
# ---------------------------------------------------------------------------
import boto3

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)


def invoke_bedrock_model(prompt: str, model_id: str = "anthropic.claude-v2") -> str:
    """Invoke an AWS Bedrock model via the bedrock-runtime API.

    Args:
        prompt: The prompt to send to the model.
        model_id: The Bedrock model ID to use.

    Returns:
        The model's response text.
    """
    import json

    body = json.dumps({"prompt": prompt, "max_tokens_to_sample": 500})
    response = bedrock_client.InvokeModel(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result.get("completion", "")


def bedrock_converse(prompt: str, model_id: str = "amazon.nova-pro-v1:0") -> str:
    """Use the Bedrock Converse API to interact with a model.

    Args:
        prompt: The user prompt.
        model_id: The Bedrock model ID.

    Returns:
        Model response text.
    """
    response = bedrock_client.Converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return response["output"]["message"]["content"][0]["text"]


# ---------------------------------------------------------------------------
# Google Vertex AI / Gemini usage
# ---------------------------------------------------------------------------
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import aiplatform
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT", "my-project"),
    location=os.getenv("GOOGLE_CLOUD_REGION", "us-central1"),
)


def chat_with_gemini(prompt: str) -> str:
    """Send a prompt to Gemini 1.5 Pro via the generativeai library.

    Args:
        prompt: The user prompt.

    Returns:
        Gemini's response text.
    """
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text


def chat_with_gemini_vertex(prompt: str) -> str:
    """Send a prompt to Gemini 1.5 Flash via Vertex AI.

    Args:
        prompt: The user prompt.

    Returns:
        Gemini Flash response text.
    """
    model = GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


# ---------------------------------------------------------------------------
# Groq usage
# ---------------------------------------------------------------------------
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def chat_with_groq(prompt: str) -> str:
    """Send a prompt to Groq's LPU inference engine.

    Args:
        prompt: The user prompt.

    Returns:
        Model response text.
    """
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Replicate usage
# ---------------------------------------------------------------------------
import replicate

replicate_api_token = os.getenv("REPLICATE_API_TOKEN")


def run_stable_diffusion(prompt: str) -> str:
    """Run Stable Diffusion on Replicate to generate an image.

    Args:
        prompt: The image generation prompt.

    Returns:
        URL of the generated image.
    """
    output = replicate.run(
        "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
        input={"prompt": prompt},
    )
    return str(output[0]) if output else ""


def run_llama_on_replicate(prompt: str) -> str:
    """Run Meta Llama on Replicate.

    Args:
        prompt: The user prompt.

    Returns:
        Generated text.
    """
    output = replicate.run(
        "meta/llama-2-70b-chat",
        input={"prompt": prompt, "max_new_tokens": 500},
    )
    return "".join(str(chunk) for chunk in output)


# ---------------------------------------------------------------------------
# Pinecone vector store usage
# ---------------------------------------------------------------------------
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def upsert_embeddings(
    index_name: str,
    vectors: list[tuple[str, list[float]]],
) -> None:
    """Upsert embedding vectors into a Pinecone index.

    Args:
        index_name: The name of the Pinecone index.
        vectors: List of (id, embedding) tuples to upsert.
    """
    index = pc.Index(index_name)
    index.upsert(vectors=[(vid, vec) for vid, vec in vectors])


def query_pinecone(
    index_name: str,
    query_vector: list[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Query a Pinecone index for similar vectors.

    Args:
        index_name: The name of the Pinecone index.
        query_vector: The query embedding vector.
        top_k: Number of nearest neighbours to return.

    Returns:
        List of matching records with scores.
    """
    index = pc.Index(index_name)
    results = index.query(vector=query_vector, top_k=top_k)
    return results.matches


# ---------------------------------------------------------------------------
# LlamaIndex usage
# ---------------------------------------------------------------------------
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.llms.openai import OpenAI as LlamaOpenAI


def build_document_index(data_dir: str) -> VectorStoreIndex:
    """Build a LlamaIndex VectorStoreIndex from a directory of documents.

    Args:
        data_dir: Path to the directory containing source documents.

    Returns:
        A populated VectorStoreIndex ready for querying.
    """
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index


def query_document_index(index: VectorStoreIndex, question: str) -> str:
    """Query a LlamaIndex document index with a natural language question.

    Args:
        index: The VectorStoreIndex to query.
        question: The question to ask.

    Returns:
        The answer text.
    """
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return str(response)


# ---------------------------------------------------------------------------
# Ollama (local LLM) usage
# ---------------------------------------------------------------------------
import ollama

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def chat_with_ollama(prompt: str, model: str = "llama3") -> str:
    """Send a prompt to a locally running Ollama model.

    Args:
        prompt: The user prompt.
        model: The Ollama model name to use.

    Returns:
        Model response text.
    """
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def pull_ollama_model(model_name: str = "mistral") -> None:
    """Pull a model from the Ollama registry.

    Args:
        model_name: The model name to pull.
    """
    ollama.pull(model_name)


# ---------------------------------------------------------------------------
# ElevenLabs text-to-speech usage
# ---------------------------------------------------------------------------
from elevenlabs import ElevenLabs, Voice

eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


def text_to_speech(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> bytes:
    """Convert text to speech using ElevenLabs.

    Args:
        text: The text to synthesise.
        voice_id: ElevenLabs voice identifier.

    Returns:
        Audio data as bytes.
    """
    audio = eleven_client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",
    )
    return b"".join(audio)


# ---------------------------------------------------------------------------
# Azure OpenAI usage
# ---------------------------------------------------------------------------
from openai import AzureOpenAI

azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
)


def chat_with_azure_openai(prompt: str) -> str:
    """Send a prompt to Azure OpenAI Service.

    Args:
        prompt: The user prompt.

    Returns:
        Model response text.
    """
    response = azure_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Together AI usage
# ---------------------------------------------------------------------------
from together import Together

together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))


def chat_with_together(prompt: str) -> str:
    """Send a prompt to Together AI's hosted open models.

    Args:
        prompt: The user prompt.

    Returns:
        Model response text.
    """
    response = together_client.chat.completions.create(
        model="meta-llama/Llama-3.1-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Weaviate vector database usage
# ---------------------------------------------------------------------------
import weaviate

weaviate_client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL", ""),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY", "")),
)


def weaviate_semantic_search(
    collection_name: str,
    query_text: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Perform a semantic search against a Weaviate collection.

    Args:
        collection_name: The name of the Weaviate collection.
        query_text: The natural language search query.
        limit: Maximum number of results to return.

    Returns:
        List of matching objects with their properties.
    """
    collection = weaviate_client.collections.get(collection_name)
    results = collection.query.near_text(query=query_text, limit=limit)
    return [obj.properties for obj in results.objects]


# ---------------------------------------------------------------------------
# AWS SageMaker usage
# ---------------------------------------------------------------------------
import sagemaker
from sagemaker import Session

sagemaker_session = Session()


def invoke_sagemaker_endpoint(endpoint_name: str, payload: str) -> str:
    """Invoke an AWS SageMaker endpoint for inference.

    Args:
        endpoint_name: The name of the SageMaker endpoint.
        payload: The JSON payload string to send.

    Returns:
        The inference response as a string.
    """
    runtime_client = boto3.client("sagemaker-runtime")
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    return response["Body"].read().decode("utf-8")


# ---------------------------------------------------------------------------
# MLflow / Databricks usage
# ---------------------------------------------------------------------------
import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def log_model_metrics(
    run_name: str,
    accuracy: float,
    loss: float,
    model_name: str = "dbrx-instruct",
) -> None:
    """Log model evaluation metrics to MLflow.

    Args:
        run_name: The MLflow run name.
        accuracy: Model accuracy score.
        loss: Model loss value.
        model_name: The model name used for the run.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)
        mlflow.log_param("model_name", model_name)


# ---------------------------------------------------------------------------
# Stability AI usage
# ---------------------------------------------------------------------------
from stability_sdk import client as stability_client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

stability_api = stability_client.StabilityInference(
    key=os.getenv("STABILITY_API_KEY", ""),
    verbose=True,
)


def generate_image_stable_diffusion(prompt: str) -> bytes:
    """Generate an image using Stability AI's Stable Diffusion API.

    Args:
        prompt: The image generation prompt.

    Returns:
        Raw PNG image bytes.
    """
    answers = stability_api.generate(
        prompt=prompt,
        steps=30,
        cfg_scale=8.0,
        width=512,
        height=512,
        samples=1,
        sampler=generation.SAMPLER_K_DPMPP_2M,
    )
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                return artifact.binary
    return b""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage demonstrating provider integrations
    print("AI Risk Scanner - Sample Project")
    print("This file demonstrates various AI provider SDK integrations.")
    print(f"OpenAI model: gpt-4o")
    print(f"Anthropic model: claude-3-5-sonnet")
    print(f"Gemini model: gemini-1.5-pro")
