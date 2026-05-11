"""Smoke test to verify environment, imports, and basic connectivity."""
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("smoke_test")

def check_env():
    """Check critical environment variables."""
    logger.info("--- Checking Environment Variables ---")
    
    # Check for Google/Gemini keys (either one is sufficient)
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key:
        masked = google_key[:4] + "*" * (len(google_key) - 8) + google_key[-4:] if len(google_key) > 8 else "***"
        logger.info(f"✅ GOOGLE_API_KEY/GEMINI_API_KEY: {masked}")
    else:
        logger.warning("❌ GOOGLE_API_KEY/GEMINI_API_KEY: Missing")

    # Other keys
    other_vars = [
        ("NVIDIA_API_KEY", True),
        ("SERPER_API_KEY", True),
        ("TAVILY_API_KEY", True),
        ("OPENAI_API_KEY", False),  # Optional
    ]
    
    all_critical_present = google_key is not None
    for var, is_critical in other_vars:
        val = os.getenv(var)
        if val:
            masked = val[:4] + "*" * (len(val) - 8) + val[-4:] if len(val) > 8 else "***"
            logger.info(f"✅ {var}: {masked}")
        else:
            if is_critical:
                logger.warning(f"❌ {var}: Missing (CRITICAL)")
                all_critical_present = False
            else:
                logger.info(f"ℹ️ {var}: Missing (Optional)")
                
    return all_critical_present

def check_imports():
    """Check if critical dependencies can be imported without errors."""
    logger.info("\n--- Checking Critical Imports ---")
    libs = [
        ("pydantic", "Pydantic (Core)"),
        ("google.genai", "Google GenAI (Gemini)"),
        ("langchain", "LangChain"),
        ("langgraph", "LangGraph (Orchestration)"),
        ("sentence_transformers", "Sentence Transformers (Embeddings)"),
        ("faiss", "FAISS (Vector Store)"),
    ]

    all_success = True
    for module_name, display_name in libs:
        try:
            __import__(module_name)
            logger.info(f"✅ {display_name}: Imported successfully")
        except ImportError as e:
            logger.error(f"❌ {display_name}: Failed to import - {e}")
            all_success = False
        except Exception as e:
            logger.error(f"❌ {display_name}: Unexpected error during import - {e}")
            all_success = False

    return all_success

def check_model_factory():
    """Check if the model factory can initialize without hitting real APIs."""
    logger.info("\n--- Checking Model Factory (Dry Run) ---")
    try:
        from trust_agents.llm.factory import create_chat_model

        # Test default config initialization
        logger.info("Initializing factory with default config...")
        # Note: We don't call the actual model if keys are missing to avoid errors,
        # but we check if the import paths work.

        provider = os.getenv("LLM_PROVIDER", "google")
        logger.info(f"Current LLM_PROVIDER: {provider}")

        # Try to create model object (will fail if API key missing, which is okay for smoke test)
        try:
            model = create_chat_model()
            logger.info(f"✅ Model factory successfully created {type(model).__name__}")
        except Exception as e:
            logger.warning(f"⚠️ Model creation failed (expected if API keys missing): {e}")

        return True
    except Exception as e:
        logger.error(f"❌ Model factory logic error: {e}")
        return False

def main():
    logger.info("=== Multi-Agent Fake News Pipeline Smoke Test ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current Directory: {Path.cwd()}")

    env_ok = check_env()
    imports_ok = check_imports()
    factory_ok = check_model_factory()

    logger.info("\n=== Summary ===")
    if env_ok and imports_ok and factory_ok:
        logger.info("✅ All core checks passed! System is ready for development.")
    elif imports_ok and factory_ok:
        logger.info("⚠️ Core code is healthy, but some API keys are missing. Real runs may fail.")
    else:
        logger.error("❌ Critical system issues detected. Please fix imports/logic before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
