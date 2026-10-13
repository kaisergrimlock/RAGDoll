#!/usr/bin/env python3
"""
Test AWS Credentials Integration
Run this to verify your AWS credentials setup is working correctly.
"""

import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports() -> bool:
    """Test that all modules can be imported."""
    try:
        from ragdoll.aws_credentials import (
            AWSCredentials,
            setup_bedrock_credentials,
            load_credentials_from_env,
            load_credentials_from_file,
            setup_from_profile,
        )
        print("✓ Module imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_credentials_creation() -> bool:
    """Test creating credentials object."""
    try:
        from ragdoll.aws_credentials import AWSCredentials
        
        creds = AWSCredentials(
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE",
            region="us-west-2",
        )
        assert creds.access_key_id == "AKIAIOSFODNN7EXAMPLE"
        assert creds.region == "us-west-2"
        print("✓ Credentials object creation works")
        return True
    except Exception as e:
        print(f"✗ Credentials creation failed: {e}")
        return False


def test_env_dict_conversion() -> bool:
    """Test converting credentials to environment dict."""
    try:
        from ragdoll.aws_credentials import AWSCredentials
        
        creds = AWSCredentials(
            access_key_id="TEST_KEY",
            secret_access_key="TEST_SECRET",
            region="eu-west-1",
        )
        env_dict = creds.to_env_dict()
        
        assert env_dict["AWS_ACCESS_KEY_ID"] == "TEST_KEY"
        assert env_dict["AWS_SECRET_ACCESS_KEY"] == "TEST_SECRET"
        assert env_dict["AWS_REGION"] == "eu-west-1"
        print("✓ Environment dictionary conversion works")
        return True
    except Exception as e:
        print(f"✗ Env dict conversion failed: {e}")
        return False


def test_setup_bedrock_credentials() -> bool:
    """Test setup_bedrock_credentials function."""
    try:
        from ragdoll.aws_credentials import setup_bedrock_credentials
        
        creds = setup_bedrock_credentials(
            access_key_id="TEST_AKIA_KEY",
            secret_access_key="test_secret_key",
            region="us-east-1",
        )
        assert creds.access_key_id == "TEST_AKIA_KEY"
        assert creds.region == "us-east-1"
        print("✓ setup_bedrock_credentials works")
        return True
    except Exception as e:
        print(f"✗ setup_bedrock_credentials failed: {e}")
        return False


def test_credentials_file_parsing() -> bool:
    """Test loading credentials from file."""
    try:
        from ragdoll.aws_credentials import load_credentials_from_file
        
        credentials_content = """[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE
region = us-west-2

[bedrock-prod]
aws_access_key_id = AKIABBBBBBBBBBBB1234
aws_secret_access_key = yJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE
region = us-east-1
"""
        
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(credentials_content)
            temp_file = f.name
        
        try:
            # Test default profile
            creds = load_credentials_from_file(temp_file, profile="default")
            assert creds is not None
            assert creds.access_key_id == "AKIAIOSFODNN7EXAMPLE"
            assert creds.region == "us-west-2"
            
            # Test bedrock-prod profile
            creds = load_credentials_from_file(temp_file, profile="bedrock-prod")
            assert creds is not None
            assert creds.access_key_id == "AKIABBBBBBBBBBBB1234"
            assert creds.region == "us-east-1"
            
            print("✓ Credentials file parsing works")
            return True
        finally:
            Path(temp_file).unlink()
    except Exception as e:
        print(f"✗ Credentials file parsing failed: {e}")
        return False


def test_env_loading() -> bool:
    """Test loading credentials from environment."""
    try:
        from ragdoll.aws_credentials import load_credentials_from_env
        
        # Save original env vars
        orig_key = os.environ.get("AWS_ACCESS_KEY_ID")
        orig_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
        orig_region = os.environ.get("AWS_REGION")
        
        try:
            # Set test credentials
            os.environ["AWS_ACCESS_KEY_ID"] = "TEST_KEY_123"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "TEST_SECRET_456"
            os.environ["AWS_REGION"] = "ap-southeast-1"
            
            creds = load_credentials_from_env()
            assert creds is not None
            assert creds.access_key_id == "TEST_KEY_123"
            assert creds.secret_access_key == "TEST_SECRET_456"
            assert creds.region == "ap-southeast-1"
            print("✓ Environment variable loading works")
            return True
        finally:
            # Restore original env vars
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
    except Exception as e:
        print(f"✗ Environment loading failed: {e}")
        return False


def main() -> int:
    """Run all tests."""
    print("\n" + "=" * 70)
    print("AWS Credentials Integration Tests")
    print("=" * 70 + "\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Credentials Creation", test_credentials_creation),
        ("Environment Dictionary Conversion", test_env_dict_conversion),
        ("Setup Bedrock Credentials", test_setup_bedrock_credentials),
        ("File Parsing", test_credentials_file_parsing),
        ("Environment Variable Loading", test_env_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing: {test_name}")
        result = test_func()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70 + "\n")
    
    if all(results):
        print("✅ All tests passed! AWS credentials integration is working.\n")
        print("Next steps:")
        print("1. Set up your AWS credentials using one of these methods:")
        print("   - setup_bedrock_credentials() for direct setup")
        print("   - setup_from_profile() for AWS CLI profiles")
        print("   - Set AWS_* environment variables")
        print("\n2. Run RAGDoll with Bedrock:")
        print("   uv run ragdoll umbrela judge \\")
        print("     --input-file examples/umbrela.requests.jsonl \\")
        print("     --output-file results/umbrela.judgments.jsonl \\")
        print("     --provider bedrock \\")
        print('     --model "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"')
        print("\n3. See docs/aws_bedrock_setup.md for full documentation.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
