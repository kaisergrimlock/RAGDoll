# AWS Bedrock Credentials Setup for RAGDoll

This guide explains how to programmatically configure AWS credentials for use with Bedrock in RAGDoll.

## Quick Start

### Option 1: Direct Credentials (Simplest)

```python
from ragdoll.aws_credentials import setup_bedrock_credentials

# Set up credentials directly in code
setup_bedrock_credentials(
    access_key_id="YOUR_AWS_ACCESS_KEY_ID",
    secret_access_key="YOUR_AWS_SECRET_ACCESS_KEY",
    region="us-west-2",
)

# Credentials are now in os.environ and ready for Bedrock
```

### Option 2: Load from AWS Profile (Recommended)

```python
from ragdoll.aws_credentials import setup_from_profile

# Load from ~/.aws/credentials (requires profile setup)
creds = setup_from_profile(profile="default")

if creds:
    print(f"Credentials loaded from region: {creds.region}")
```

### Option 3: Environment Variables

```python
from ragdoll.aws_credentials import load_credentials_from_env

# Load from AWS_* environment variables
creds = load_credentials_from_env()

if creds:
    print(f"Credentials loaded: {creds.region}")
```

## Setup Methods

### Setting Up AWS CLI Profiles

Create or edit `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE
region = us-west-2

[bedrock-prod]
aws_access_key_id = AKIABBBBBBBBBBBB1234
aws_secret_access_key = yJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE
region = us-east-1

[bedrock-dev]
aws_access_key_id = AKIACCCCCCCCCCCC5678
aws_secret_access_key = zJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE
region = eu-west-1
```

Then load with:
```python
setup_from_profile(profile="bedrock-prod")
```

### Using .env File

Create `.env` in your RAGDoll directory:

```bash
AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"
AWS_REGION="us-west-2"
AWS_SESSION_TOKEN="YOUR_SESSION_TOKEN"  # Optional
```

Then load with:
```python
load_credentials_from_env()
```

## Complete Example: Running Evaluation with Bedrock

```python
from pathlib import Path
from ragdoll.aws_credentials import setup_from_profile
from ragdoll.config import RunConfig

# Step 1: Set up credentials
creds = setup_from_profile(profile="bedrock-eval")
if not creds:
    raise SystemExit("Could not load bedrock-eval profile")

print(f"✓ AWS credentials loaded for region {creds.region}")

# Step 2: Create RAGDoll config
config = RunConfig(
    input_file=Path("examples/umbrela.requests.jsonl"),
    output_file=Path("results/umbrela.judgments.jsonl"),
    provider="bedrock",
    model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    raw_events_dir=Path("results/umbrela.raw-events"),
    max_concurrency=2,
    overwrite=True,
)

print(f"✓ RAGDoll config ready")
print(f"  Provider: {config.provider}")
print(f"  Model: {config.model}")

# Step 3: Run evaluation via CLI
# uv run ragdoll umbrela judge \
#   --input-file examples/umbrela.requests.jsonl \
#   --output-file results/umbrela.judgments.jsonl \
#   --provider bedrock \
#   --model "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
```

## API Reference

### `AWSCredentials`

Dataclass representing AWS credentials:

```python
from ragdoll.aws_credentials import AWSCredentials

creds = AWSCredentials(
    access_key_id="AKIA...",
    secret_access_key="wJalr...",
    region="us-west-2",
    session_token=None,  # Optional, for temporary credentials
)

# Convert to environment variable dictionary
env_dict = creds.to_env_dict()
# {'AWS_ACCESS_KEY_ID': '...', 'AWS_SECRET_ACCESS_KEY': '...', 'AWS_REGION': '...'}

# Inject directly into os.environ
creds.inject_env()
```

### `setup_bedrock_credentials()`

Set up credentials directly in code:

```python
from ragdoll.aws_credentials import setup_bedrock_credentials

creds = setup_bedrock_credentials(
    access_key_id="YOUR_KEY",
    secret_access_key="YOUR_SECRET",
    region="us-west-2",
    session_token=None,  # Optional
)
```

### `setup_from_profile()`

Load credentials from AWS CLI profile:

```python
from ragdoll.aws_credentials import setup_from_profile

creds = setup_from_profile(
    profile="default",
    credentials_file=None,  # Uses ~/.aws/credentials if None
)

if creds:
    print(f"Region: {creds.region}")
```

### `load_credentials_from_env()`

Load credentials from environment variables:

```python
from ragdoll.aws_credentials import load_credentials_from_env

creds = load_credentials_from_env()
# Looks for: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, AWS_SESSION_TOKEN
```

### `load_credentials_from_file()`

Load credentials from AWS credentials file:

```python
from ragdoll.aws_credentials import load_credentials_from_file
from pathlib import Path

creds = load_credentials_from_file(
    credentials_file=Path.home() / ".aws" / "credentials",
    profile="bedrock-prod",
)
```

## Temporary Credentials (STS)

For temporary credentials from AWS STS:

```python
from ragdoll.aws_credentials import setup_bedrock_credentials

# When using STS temporary credentials
creds = setup_bedrock_credentials(
    access_key_id="ASIA...",  # Note: starts with ASIA for temporary
    secret_access_key="temporary_secret_key",
    region="us-west-2",
    session_token="FwoGZXIvYXdzE...",  # Required for temporary credentials
)
```

## Bedrock Model IDs

Common Bedrock models available for use:

```python
# Claude 3.5 Sonnet (Recommended)
model = "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"

# Claude 3 Opus
model = "bedrock/anthropic.claude-3-opus-20250219-v1:0"

# Claude 3 Haiku
model = "bedrock/anthropic.claude-3-haiku-20250307-v1:0"

# Llama 2
model = "bedrock/meta.llama2-13b-chat-v1"
```

## Security Best Practices

1. **Never hardcode credentials in source code** - Use profiles or environment variables
2. **Use IAM roles** when running on AWS infrastructure (EC2, Lambda, ECS)
3. **Rotate credentials** regularly
4. **Use temporary credentials (STS)** for short-lived access
5. **Restrict IAM permissions** to only Bedrock InvokeModel API calls

Example minimal IAM policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": "arn:aws:bedrock:*:ACCOUNT:foundation-model/*"
        }
    ]
}
```

## Running Tests

```bash
# Install test dependencies
uv sync --extra dev

# Run credential tests
uv run pytest tests/test_aws_credentials.py -v

# Run all tests
uv run pytest
```

## Troubleshooting

### Credentials not found
```python
from ragdoll.aws_credentials import load_credentials_from_env

creds = load_credentials_from_env()
if not creds:
    print("AWS credentials not found in environment")
    print("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
```

### Profile not found
```python
from ragdoll.aws_credentials import setup_from_profile

creds = setup_from_profile(profile="bedrock-prod")
if not creds:
    print("Profile 'bedrock-prod' not found")
    print(f"Check ~/.aws/credentials")
```

### Bedrock API errors
- Ensure AWS region supports Bedrock
- Check IAM permissions for bedrock:InvokeModel
- Verify model ID is correct for your region
- Check model is not in limited access

## Examples

See [examples/setup_bedrock_credentials.py](examples/setup_bedrock_credentials.py) for complete working examples.
