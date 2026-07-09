"""Tests for AWS credential management."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from ragdoll.aws_credentials import (
    AWSCredentials,
    load_credentials_from_env,
    load_credentials_from_file,
    setup_bedrock_credentials,
    setup_from_profile,
)


class TestAWSCredentials:
    """Tests for AWSCredentials dataclass."""

    def test_credentials_creation(self) -> None:
        """Test basic credential creation."""
        creds = AWSCredentials(
            access_key_id="AKIA123456789ABCDEF",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE",
            region="us-west-2",
        )
        assert creds.access_key_id == "AKIA123456789ABCDEF"
        assert creds.secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE"
        assert creds.region == "us-west-2"
        assert creds.session_token is None

    def test_to_env_dict(self) -> None:
        """Test conversion to environment variable dictionary."""
        creds = AWSCredentials(
            access_key_id="AKIA123456789ABCDEF",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE",
            region="eu-west-1",
        )
        env_dict = creds.to_env_dict()
        
        assert env_dict["AWS_ACCESS_KEY_ID"] == "AKIA123456789ABCDEF"
        assert env_dict["AWS_SECRET_ACCESS_KEY"] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE"
        assert env_dict["AWS_REGION"] == "eu-west-1"
        assert "AWS_SESSION_TOKEN" not in env_dict

    def test_to_env_dict_with_session_token(self) -> None:
        """Test conversion with temporary session token."""
        creds = AWSCredentials(
            access_key_id="ASIA123456789ABCDEF",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE",
            region="us-east-1",
            session_token="FwoGZXIvYXdzEBaaDCMTJ7EXAMPLE123/ABC...",
        )
        env_dict = creds.to_env_dict()
        
        assert env_dict["AWS_SESSION_TOKEN"] == "FwoGZXIvYXdzEBaaDCMTJ7EXAMPLE123/ABC..."

    def test_inject_env(self) -> None:
        """Test injecting credentials into os.environ."""
        # Save original values
        orig_key = os.environ.get("AWS_ACCESS_KEY_ID")
        orig_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
        orig_region = os.environ.get("AWS_REGION")
        
        try:
            creds = AWSCredentials(
                access_key_id="TEST_KEY_123",
                secret_access_key="TEST_SECRET_456",
                region="ap-southeast-1",
            )
            creds.inject_env()
            
            assert os.environ["AWS_ACCESS_KEY_ID"] == "TEST_KEY_123"
            assert os.environ["AWS_SECRET_ACCESS_KEY"] == "TEST_SECRET_456"
            assert os.environ["AWS_REGION"] == "ap-southeast-1"
        finally:
            # Restore original values
            if orig_key is not None:
                os.environ["AWS_ACCESS_KEY_ID"] = orig_key
            elif "AWS_ACCESS_KEY_ID" in os.environ:
                del os.environ["AWS_ACCESS_KEY_ID"]
            
            if orig_secret is not None:
                os.environ["AWS_SECRET_ACCESS_KEY"] = orig_secret
            elif "AWS_SECRET_ACCESS_KEY" in os.environ:
                del os.environ["AWS_SECRET_ACCESS_KEY"]
            
            if orig_region is not None:
                os.environ["AWS_REGION"] = orig_region
            elif "AWS_REGION" in os.environ:
                del os.environ["AWS_REGION"]


class TestLoadCredentialsFromEnv:
    """Tests for loading credentials from environment variables."""

    def test_load_complete_credentials(self) -> None:
        """Test loading complete credentials from environment."""
        orig_key = os.environ.get("AWS_ACCESS_KEY_ID")
        orig_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
        orig_region = os.environ.get("AWS_REGION")
        
        try:
            os.environ["AWS_ACCESS_KEY_ID"] = "AKIA_TEST_KEY"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_key"
            os.environ["AWS_REGION"] = "us-west-2"
            
            creds = load_credentials_from_env()
            
            assert creds is not None
            assert creds.access_key_id == "AKIA_TEST_KEY"
            assert creds.secret_access_key == "test_secret_key"
            assert creds.region == "us-west-2"
        finally:
            if orig_key is not None:
                os.environ["AWS_ACCESS_KEY_ID"] = orig_key
            elif "AWS_ACCESS_KEY_ID" in os.environ:
                del os.environ["AWS_ACCESS_KEY_ID"]
            
            if orig_secret is not None:
                os.environ["AWS_SECRET_ACCESS_KEY"] = orig_secret
            elif "AWS_SECRET_ACCESS_KEY" in os.environ:
                del os.environ["AWS_SECRET_ACCESS_KEY"]
            
            if orig_region is not None:
                os.environ["AWS_REGION"] = orig_region
            elif "AWS_REGION" in os.environ:
                del os.environ["AWS_REGION"]

    def test_load_missing_credentials(self) -> None:
        """Test handling of missing credentials."""
        orig_key = os.environ.pop("AWS_ACCESS_KEY_ID", None)
        orig_secret = os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
        
        try:
            creds = load_credentials_from_env()
            assert creds is None
        finally:
            if orig_key is not None:
                os.environ["AWS_ACCESS_KEY_ID"] = orig_key
            if orig_secret is not None:
                os.environ["AWS_SECRET_ACCESS_KEY"] = orig_secret


class TestLoadCredentialsFromFile:
    """Tests for loading credentials from AWS credentials file."""

    def test_load_from_credentials_file(self) -> None:
        """Test loading credentials from a file."""
        content = """[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE
region = us-west-2

[bedrock-prod]
aws_access_key_id = AKIABBBBBBBBBBBB1234
aws_secret_access_key = yJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE
region = us-east-1
"""
        
        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            # Test loading default profile
            creds = load_credentials_from_file(temp_path, profile="default")
            assert creds is not None
            assert creds.access_key_id == "AKIAIOSFODNN7EXAMPLE"
            assert creds.region == "us-west-2"
            
            # Test loading bedrock-prod profile
            creds = load_credentials_from_file(temp_path, profile="bedrock-prod")
            assert creds is not None
            assert creds.access_key_id == "AKIABBBBBBBBBBBB1234"
            assert creds.region == "us-east-1"
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_profile(self) -> None:
        """Test loading non-existent profile."""
        content = "[default]\naws_access_key_id = TEST\naws_secret_access_key = TEST\n"
        
        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        try:
            creds = load_credentials_from_file(temp_path, profile="nonexistent")
            assert creds is None
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self) -> None:
        """Test loading from non-existent file."""
        creds = load_credentials_from_file(Path("/nonexistent/path/credentials"))
        assert creds is None


class TestSetupBedrockcredentials:
    """Tests for setup_bedrock_credentials function."""

    def test_setup_credentials(self) -> None:
        """Test setting up credentials."""
        orig_key = os.environ.get("AWS_ACCESS_KEY_ID")
        orig_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
        
        try:
            creds = setup_bedrock_credentials(
                access_key_id="TEST_KEY",
                secret_access_key="TEST_SECRET",
                region="eu-central-1",
            )
            
            assert creds.access_key_id == "TEST_KEY"
            assert creds.secret_access_key == "TEST_SECRET"
            assert creds.region == "eu-central-1"
            assert os.environ["AWS_ACCESS_KEY_ID"] == "TEST_KEY"
            assert os.environ["AWS_SECRET_ACCESS_KEY"] == "TEST_SECRET"
        finally:
            if orig_key is not None:
                os.environ["AWS_ACCESS_KEY_ID"] = orig_key
            elif "AWS_ACCESS_KEY_ID" in os.environ:
                del os.environ["AWS_ACCESS_KEY_ID"]
            
            if orig_secret is not None:
                os.environ["AWS_SECRET_ACCESS_KEY"] = orig_secret
            elif "AWS_SECRET_ACCESS_KEY" in os.environ:
                del os.environ["AWS_SECRET_ACCESS_KEY"]
