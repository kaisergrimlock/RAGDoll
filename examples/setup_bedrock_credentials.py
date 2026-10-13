#!/usr/bin/env python3
"""Example: Set up AWS credentials and run Bedrock evaluation with RAGDoll.

This script demonstrates how to programmatically configure AWS credentials
for use with Bedrock and RAGDoll's evaluation workflows.
"""

from pathlib import Path

from ragdoll.aws_credentials import setup_bedrock_credentials, setup_from_profile
from ragdoll.config import RunConfig
from ragdoll.runner import run_task_rows


def example_direct_credentials():
    """Example 1: Use credentials directly in code."""
    print("Example 1: Direct credential setup")
    print("-" * 50)
    
    # Set up credentials directly
    creds = setup_bedrock_credentials(
        access_key_id="YOUR_AWS_ACCESS_KEY_ID",
        secret_access_key="YOUR_AWS_SECRET_ACCESS_KEY",
        region="us-east-1",
    )
    print(f"✓ Credentials set up for region: {creds.region}")
    print()


def example_profile_credentials():
    """Example 2: Load from AWS CLI profile."""
    print("Example 2: Load from AWS profile")
    print("-" * 50)
    
    # Load from default AWS profile
    creds = setup_from_profile(profile="default")
    if creds:
        print(f"✓ Loaded default profile from {creds.region}")
    else:
        print("✗ Could not load default profile")
    
    # Or load a specific profile
    creds = setup_from_profile(profile="bedrock-prod")
    if creds:
        print(f"✓ Loaded bedrock-prod profile from {creds.region}")
    print()


def example_with_ragdoll_config():
    """Example 3: Use credentials with RAGDoll config."""
    print("Example 3: Configure RAGDoll with Bedrock")
    print("-" * 50)
    
    # Set up credentials first
    setup_bedrock_credentials(
        access_key_id="YOUR_AWS_ACCESS_KEY_ID",
        secret_access_key="YOUR_AWS_SECRET_ACCESS_KEY",
        region="us-west-2",
    )
    
    # Create RAGDoll config for Bedrock
    config = RunConfig(
        input_file=Path("examples/umbrela.requests.jsonl"),
        output_file=Path("results/umbrela.judgments.jsonl"),
        provider="bedrock",
        model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        thinking="minimal",
        max_concurrency=4,
        raw_events_dir=Path("results/umbrela.raw-events"),
    )
    
    print(f"✓ RAGDoll config created")
    print(f"  Provider: {config.provider}")
    print(f"  Model: {config.model}")
    print(f"  Input: {config.input_file}")
    print()


def example_with_env_fallback():
    """Example 4: Fallback to environment variables."""
    import os
    from ragdoll.aws_credentials import load_credentials_from_env
    
    print("Example 4: Load from environment variables")
    print("-" * 50)
    
    # Try to load from environment first
    creds = load_credentials_from_env()
    if creds:
        print(f"✓ Loaded credentials from environment")
        print(f"  Region: {creds.region}")
    else:
        print("✗ No AWS credentials found in environment")
        print("  Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    print()


def example_full_workflow():
    """Example 5: Complete workflow for UMBRELA evaluation."""
    print("Example 5: Full UMBRELA evaluation workflow")
    print("-" * 50)
    
    # 1. Set up credentials
    print("Step 1: Setting up AWS credentials...")
    creds = setup_from_profile(profile="bedrock-eval")
    if not creds:
        print("Error: Could not load bedrock-eval profile")
        print("Expected file: ~/.aws/credentials")
        print("[bedrock-eval]")
        print("aws_access_key_id = YOUR_KEY")
        print("aws_secret_access_key = YOUR_SECRET")
        print("region = us-west-2")
        return
    
    print(f"✓ Credentials loaded from bedrock-eval profile")
    
    # 2. Create config
    print("\nStep 2: Creating RAGDoll config...")
    config = RunConfig(
        input_file=Path("examples/umbrela.requests.jsonl"),
        output_file=Path("results/umbrela.judgments.jsonl"),
        provider="bedrock",
        model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        raw_events_dir=Path("results/umbrela.raw-events"),
        cache_dir=Path(".cache/ragdoll"),
        max_concurrency=2,
        overwrite=True,
    )
    print(f"✓ Config ready")
    
    # 3. Print command to run
    print("\nStep 3: To run the evaluation, execute:")
    print("-" * 50)
    print("uv run ragdoll umbrela judge \\")
    print(f"  --input-file {config.input_file} \\")
    print(f"  --output-file {config.output_file} \\")
    print(f"  --provider {config.provider} \\")
    print(f"  --model {config.model}")
    print("-" * 50)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("AWS Credentials Setup Examples for RAGDoll + Bedrock")
    print("=" * 50 + "\n")
    
    example_direct_credentials()
    example_profile_credentials()
    example_with_ragdoll_config()
    example_with_env_fallback()
    example_full_workflow()
    
    print("\n" + "=" * 50)
    print("For production use:")
    print("1. Set up ~/.aws/credentials with your profile")
    print("2. Use setup_from_profile() to load credentials")
    print("3. Configure RAGDoll with provider='bedrock'")
    print("=" * 50 + "\n")
