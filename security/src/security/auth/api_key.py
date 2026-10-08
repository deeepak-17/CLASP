from __future__ import annotations

import hashlib
import hmac
import secrets
import string


def generate_api_key(prefix: str = "clasp") -> str:
    """
    Generate a URL-safe random API key with a given prefix.
    
    Args:
        prefix: The prefix string for the API key. Defaults to 'clasp'.
        
    Returns:
        The generated API key string in the format '{prefix}_{32 random chars}'.
    """
    # Use URL-safe characters for the random part
    alphabet = string.ascii_letters + string.digits + "-_"
    random_part = "".join(secrets.choice(alphabet) for _ in range(32))
    return f"{prefix}_{random_part}"


def hash_api_key(key: str) -> str:
    """
    SHA-256 hash an API key for safe storage.
    
    Args:
        key: The raw API key.
        
    Returns:
        The hex-encoded SHA-256 hash of the API key.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def verify_api_key(key: str, stored_hash: str) -> bool:
    """
    Verify a key against its stored hash using constant-time comparison.
    
    Args:
        key: The raw API key to verify.
        stored_hash: The previously computed and stored hash of the key.
        
    Returns:
        True if the key matches the stored hash, False otherwise.
    """
    computed_hash = hash_api_key(key)
    return hmac.compare_digest(computed_hash, stored_hash)


class APIKeyStore:
    """
    In-memory store mapping service names to hashed API keys.
    Intended for simple service authentication within the system.
    """
    
    def __init__(self) -> None:
        """Initialize an empty in-memory API key store."""
        self._store: dict[str, str] = {}

    def register(self, service_name: str) -> str:
        """
        Generate and store an API key for a given service.
        
        Args:
            service_name: The name of the service to register.
            
        Returns:
            The raw generated API key (this is the only time it will be returned).
        """
        raw_key = generate_api_key()
        self._store[service_name] = hash_api_key(raw_key)
        return raw_key

    def authenticate(self, service_name: str, key: str) -> bool:
        """
        Authenticate a service by verifying the provided API key.
        
        Args:
            service_name: The name of the service attempting to authenticate.
            key: The raw API key provided by the service.
            
        Returns:
            True if the service is registered and the key is valid, False otherwise.
        """
        stored_hash = self._store.get(service_name)
        if not stored_hash:
            return False
            
        return verify_api_key(key, stored_hash)

    def revoke(self, service_name: str) -> None:
        """
        Revoke the API key for a given service.
        
        Args:
            service_name: The name of the service whose key should be revoked.
        """
        self._store.pop(service_name, None)

    def list_services(self) -> list[str]:
        """
        List all registered service names.
        
        Returns:
            A list of service names currently in the store.
        """
        return list(self._store.keys())
