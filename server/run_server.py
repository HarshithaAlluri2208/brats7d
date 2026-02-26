"""
Server startup script for NeuroVision Inference Server.

This script:
- Checks Python and dependency versions
- Configures uvicorn settings
- Runs the FastAPI server programmatically
"""
import os
import sys
import uvicorn

def print_versions():
    """Print Python and key dependency versions."""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: Not installed")
    
    try:
        import monai
        print(f"MONAI version: {monai.__version__}")
    except ImportError:
        print("MONAI: Not installed")
    
    try:
        import fastapi
        print(f"FastAPI version: {fastapi.__version__}")
    except ImportError:
        print("FastAPI: Not installed")
    
    try:
        import uvicorn
        print(f"Uvicorn version: {uvicorn.__version__}")
    except ImportError:
        print("Uvicorn: Not installed")


def main():
    """Main entry point for server startup."""
    print("=" * 60)
    print("NeuroVision Inference Server - Starting...")
    print("=" * 60)
    print()
    
    # Print versions
    print_versions()
    print()
    
    # Get configuration from environment or use defaults
    host = os.getenv("UVICORN_HOST", "0.0.0.0")
    port = int(os.getenv("UVICORN_PORT", "8000"))
    reload = os.getenv("UVICORN_RELOAD", "true").lower() == "true"
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info")
    
    # Check for custom uvicorn command in env
    uvicorn_cmd = os.getenv("UVICORN_CMD")
    if uvicorn_cmd:
        print(f"Using custom UVICORN_CMD: {uvicorn_cmd}")
        # Parse and execute custom command if provided
        # For now, we'll use the programmatic approach
    
    print(f"Server configuration:")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Reload: {reload}")
    print(f"  Log level: {log_level}")
    print()
    print("=" * 60)
    print(f"Starting server on http://{host}:{port}")
    print("=" * 60)
    print()
    
    # Run uvicorn programmatically
    try:
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

