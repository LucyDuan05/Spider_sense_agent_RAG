"""
Spider-Sense v2 — Main Entry Point
Double-click to launch the full IDS system.

Usage:
    python main.py              # Start server
    python main.py --port 8080  # Custom port
    python main.py --debug      # Debug mode
"""
import os
import sys
import argparse
import webbrowser
import threading
import time


def open_browser(url: str, delay: float = 1.5):
    """Open browser after a short delay to let Flask start."""
    time.sleep(delay)
    print(f"[*] Opening browser at {url}")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(
        description='Spider-Sense v2 — Multi-Agent Open-Set Network IDS'
    )
    parser.add_argument('--port', type=int, default=5000, help='Server port (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Do not auto-open browser')
    parser.add_argument('--llm-api-key', type=str, default=None, help='LLM API key for agent mode')
    parser.add_argument('--llm-model', type=str, default='gpt-4', help='LLM model name')
    parser.add_argument('--llm-api-base', type=str, default=None, help='LLM API base URL')
    args = parser.parse_args()

    # Set environment for submodules
    if args.llm_api_key:
        os.environ['LLM_API_KEY'] = args.llm_api_key
    if args.llm_model:
        os.environ['LLM_MODEL'] = args.llm_model
    if args.llm_api_base:
        os.environ['LLM_API_BASE'] = args.llm_api_base

    os.environ['PORT'] = str(args.port)
    if args.debug:
        os.environ['DEBUG'] = 'true'

    # Print banner (use ASCII-safe characters for Windows compatibility)
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   Spider-Sense v2  --  Open-Set Network IDS                  ║
    ║                                                              ║
    ║   DHRNet-1D + Weibull OpenMax + Semantic RAG + XAI           ║
    ║                                                              ║
    ║   Version: 2.0.0                                             ║
    ║   Model: DHRNet-1D + Weibull OpenMax (pure Python)           ║
    ║   Agent: Senior SecOps Analyst (Single-Agent)                ║
    ║   Knowledge: MITRE ATT&CK Semantic RAG                       ║
    ║   XAI: Feature Perturbation Attribution                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    url = f'http://{args.host}:{args.port}'

    # Auto-open browser
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    # Import and run Flask app
    from backend.app import app

    print(f"""
    ┌──────────────────────────────────────────────────────────┐
    │  Server:  {url:<50}│
    │  API:     {url}/api/health{'':<43}│
    │  Mode:    {'LLM Agent' if args.llm_api_key else 'Rule-Based Agent':<51}│
    └──────────────────────────────────────────────────────────┘
    """)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
