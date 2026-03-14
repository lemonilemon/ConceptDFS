import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, List
from rich.console import Console
from rich.prompt import Prompt
from concept_dfs.paths import DATA_DIR

console = Console()

AUTH_FILE = DATA_DIR / "auth.json"


class MissingAPIKeyError(Exception):
    """Raised when no API key can be resolved for a provider."""

    def __init__(self, provider_id: str, env_key: str):
        self.provider_id = provider_id
        self.env_key = env_key
        super().__init__(
            f"No API key found for provider '{provider_id}'. "
            f"Set the {env_key} environment variable or run: concept-dfs auth"
        )


@dataclass
class ProviderInfo:
    """Metadata for a supported LLM provider."""

    name: str
    env_key: str
    base_url: str
    default_model: str


PROVIDERS: Dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        name="OpenAI",
        env_key="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4.1",
    ),
    "openrouter": ProviderInfo(
        name="OpenRouter",
        env_key="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-2.5-pro",
    ),
    "gemini": ProviderInfo(
        name="Google Gemini",
        env_key="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.5-pro",
    ),
    "anthropic": ProviderInfo(
        name="Anthropic (via OpenRouter)",
        env_key="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-sonnet-4-20250514",
    ),
}


def _load_auth() -> dict:
    """Load saved credentials from auth.json."""
    if AUTH_FILE.exists():
        try:
            return json.loads(AUTH_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_auth(data: dict) -> None:
    """Write credentials to auth.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(json.dumps(data, indent=2) + "\n")
    AUTH_FILE.chmod(0o600)


def get_api_key(provider_id: str) -> Optional[str]:
    """Resolve API key: env var > auth.json > None."""
    info = PROVIDERS.get(provider_id)
    if not info:
        return None

    # 1. Check environment variable
    key = os.environ.get(info.env_key)
    if key:
        return key

    # 2. Check saved credentials
    auth = _load_auth()
    return auth.get(provider_id)


def save_api_key(provider_id: str, key: str) -> None:
    """Persist an API key to auth.json."""
    auth = _load_auth()
    auth[provider_id] = key
    _save_auth(auth)


def ensure_api_key(provider_id: str) -> str:
    """Ensure an API key is available for the given provider.

    Returns the API key if found (env var or saved auth).
    Raises MissingAPIKeyError if no key is available.
    Raises ValueError for unknown providers.
    """
    info = PROVIDERS.get(provider_id)
    if not info:
        raise ValueError(
            f"Unknown provider: {provider_id}. Supported: {', '.join(PROVIDERS.keys())}"
        )

    key = get_api_key(provider_id)

    if not key:
        raise MissingAPIKeyError(provider_id, info.env_key)

    return key


def force_auth(provider_id: Optional[str] = None) -> None:
    """Force interactive prompt for API key, overriding any saved key."""
    if not provider_id:
        model_str = resolve_model()
        provider_id = model_str.split(":")[0] if ":" in model_str else model_str

    info = PROVIDERS.get(provider_id)
    if not info:
        console.print(f"[bold red]Unknown provider: {provider_id}[/bold red]")
        return

    console.print(f"\n[bold cyan]Configuring API key for {info.name}[/bold cyan]")
    key = Prompt.ask(f"[bold cyan]{info.env_key}[/bold cyan]").strip()

    if key:
        save_api_key(provider_id, key)
        console.print(f"[bold green]Key saved to {AUTH_FILE}[/bold green]\n")
        os.environ[info.env_key] = key
    else:
        console.print("[dim]No key provided. Cancelled.[/dim]\n")


def get_saved_model() -> Optional[str]:
    """Get saved model preference from auth.json."""
    auth = _load_auth()
    return auth.get("_model")


def save_model(model_str: str) -> None:
    """Persist model preference to auth.json."""
    auth = _load_auth()
    auth["_model"] = model_str
    _save_auth(auth)


def resolve_model() -> str:
    """Resolve the model string: env var > saved config > interactive selection."""
    # 1. Check environment variable
    env_model = os.environ.get("CONCEPT_DFS_MODEL")
    if env_model:
        return env_model

    # 2. Check saved preference
    saved = get_saved_model()
    if saved:
        return saved

    # 3. Interactive selection
    return select_model()


def select_model() -> str:
    """Interactive model picker using a numbered menu.

    Returns a 'provider:model' string and saves the choice.
    """
    console.print("\n[bold cyan]Select a provider:[/bold cyan]")

    provider_ids = list(PROVIDERS.keys())
    for i, info in enumerate(PROVIDERS.values(), 1):
        console.print(
            f"  [bold green]{i}.[/bold green] {info.name}  ({info.default_model})"
        )

    while True:
        choice = Prompt.ask("\n[bold]Provider number[/bold]").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(provider_ids):
                break
        console.print("[red]Invalid choice. Try again.[/red]")

    provider_id = provider_ids[idx]
    info = PROVIDERS[provider_id]

    console.print(f"\n[dim]Default model: [bold]{info.default_model}[/bold][/dim]")
    custom = Prompt.ask(
        "[bold]Model name[/bold] (press Enter for default)",
        default=info.default_model,
    )
    model = custom.strip() or info.default_model

    model_str = f"{provider_id}:{model}"
    save_model(model_str)
    console.print(f"[bold green]Model set to {model_str}[/bold green]\n")
    return model_str
