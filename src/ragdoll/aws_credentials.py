"""AWS credential management for Bedrock integration.

Provides utilities to configure AWS credentials programmatically for use with
the Bedrock provider. Supports multiple authentication methods:
- Direct key configuration
- Profile-based authentication
- Environment variable injection
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AWSCredentials:
    """AWS credentials for Bedrock authentication."""

    access_key_id: str
    secret_access_key: str
    region: str = "us-west-2"
    session_token: Optional[str] = None

    def to_env_dict(self) -> dict[str, str]:
        """Convert credentials to environment variable dictionary.
        
        Returns:
            Dictionary with AWS_* environment variables.
        """
        env = {
            "AWS_ACCESS_KEY_ID": self.access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.secret_access_key,
            "AWS_REGION": self.region,
        }
        if self.session_token:
            env["AWS_SESSION_TOKEN"] = self.session_token
        return env

    def inject_env(self) -> None:
        """Inject credentials directly into os.environ.
        
        This modifies the current process's environment. Useful when
        credentials need to be available before subprocess spawning.
        """
        env_dict = self.to_env_dict()
        os.environ.update(env_dict)


def load_credentials_from_env() -> AWSCredentials | None:
    """Load AWS credentials from environment variables.
    
    Looks for standard AWS environment variables:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_REGION (defaults to us-west-2)
    - AWS_SESSION_TOKEN (optional, for temporary credentials)
    
    Returns:
        AWSCredentials if all required variables are set, None otherwise.
    """
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    
    if not (access_key and secret_key):
        return None
    
    return AWSCredentials(
        access_key_id=access_key,
        secret_access_key=secret_key,
        region=os.environ.get("AWS_REGION", "us-west-2").strip(),
        session_token=os.environ.get("AWS_SESSION_TOKEN", "").strip() or None,
    )


def load_credentials_from_file(
    credentials_file: Path | str,
    profile: str = "default",
) -> AWSCredentials | None:
    """Load AWS credentials from ~/.aws/credentials file.
    
    Args:
        credentials_file: Path to credentials file (usually ~/.aws/credentials)
        profile: AWS profile name to load (default: "default")
    
    Returns:
        AWSCredentials if profile is found and complete, None otherwise.
        
    Example:
        >>> creds = load_credentials_from_file(Path.home() / ".aws" / "credentials")
        >>> if creds:
        ...     creds.inject_env()
    """
    credentials_file = Path(credentials_file)
    if not credentials_file.is_file():
        return None
    
    try:
        content = credentials_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    
    profile_section = f"[{profile}]"
    if profile_section not in content:
        return None
    
    # Parse the profile section
    lines = content.split("\n")
    in_profile = False
    creds_dict: dict[str, str] = {}
    
    for line in lines:
        line = line.strip()
        
        # Start of profile section
        if line == profile_section:
            in_profile = True
            continue
        
        # End of profile section (start of another)
        if in_profile and line.startswith("[") and line.endswith("]"):
            break
        
        if not in_profile:
            continue
        
        # Skip comments and empty lines
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        
        # Parse key=value
        if "=" in line:
            key, value = line.split("=", 1)
            creds_dict[key.strip()] = value.strip()
    
    # Extract required fields
    access_key = creds_dict.get("aws_access_key_id", "").strip()
    secret_key = creds_dict.get("aws_secret_access_key", "").strip()
    
    if not (access_key and secret_key):
        return None
    
    return AWSCredentials(
        access_key_id=access_key,
        secret_access_key=secret_key,
        region=creds_dict.get("region", "us-west-2").strip() or "us-west-2",
        session_token=creds_dict.get("aws_session_token", "").strip() or None,
    )


def setup_bedrock_credentials(
    access_key_id: str,
    secret_access_key: str,
    region: str = "us-west-2",
    session_token: str | None = None,
) -> AWSCredentials:
    """Set up Bedrock credentials programmatically.
    
    Creates and injects AWS credentials into the environment for use with
    Bedrock. This is the recommended way to configure credentials in code
    rather than using environment variables directly.
    
    Args:
        access_key_id: AWS Access Key ID
        secret_access_key: AWS Secret Access Key
        region: AWS region (default: us-west-2)
        session_token: Optional session token for temporary credentials
    
    Returns:
        AWSCredentials object with injected environment variables
        
    Example:
        >>> creds = setup_bedrock_credentials(
        ...     access_key_id="AKIAIOSFODNN7EXAMPLE",
        ...     secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        ...     region="us-east-1"
        ... )
        >>> # Credentials are now in os.environ and ready for Bedrock
    """
    creds = AWSCredentials(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        region=region,
        session_token=session_token,
    )
    creds.inject_env()
    return creds


def setup_from_profile(
    profile: str = "default",
    credentials_file: Path | str | None = None,
) -> AWSCredentials | None:
    """Set up credentials from an AWS CLI profile.
    
    Loads credentials from ~/.aws/credentials by default, or a custom path.
    Automatically injects them into os.environ.
    
    Args:
        profile: AWS profile name (default: "default")
        credentials_file: Custom path to credentials file
    
    Returns:
        AWSCredentials if profile found and loaded, None otherwise
        
    Example:
        >>> creds = setup_from_profile(profile="my-bedrock-profile")
        >>> if creds:
        ...     print(f"Loaded profile: {creds.region}")
    """
    if credentials_file is None:
        credentials_file = Path.home() / ".aws" / "credentials"
    
    creds = load_credentials_from_file(credentials_file, profile=profile)
    if creds:
        creds.inject_env()
    return creds
