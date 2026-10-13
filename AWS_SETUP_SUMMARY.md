# AWS Bedrock Integration - Setup Guide

You've successfully set up AWS credential management for RAGDoll + Bedrock! Here's what's been added:

## 📦 What Was Created

### Core Module
- **`src/ragdoll/aws_credentials.py`** - AWS credential management module
  - `setup_bedrock_credentials()` - Direct credential setup
  - `setup_from_profile()` - Load from AWS CLI profile
  - `load_credentials_from_env()` - Load from environment variables
  - `load_credentials_from_file()` - Load from custom credentials file
  - `AWSCredentials` class - Credential data structure

### Examples & Quick Start
- **`examples/setup_bedrock_credentials.py`** - Complete working examples
- **`scripts/aws_credentials_quickstart.py`** - Quick start guide
- **`scripts/validate_aws_credentials.py`** - Validation script

### Documentation
- **`docs/aws_bedrock_setup.md`** - Full documentation with API reference
- **`.env.example`** - Updated with AWS credential options

### Tests
- **`tests/test_aws_credentials.py`** - Comprehensive test suite

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd D:\Work\RAGDoll
uv sync
```

### 2. Choose a Credential Setup Method

#### Method A: Direct Setup (Simplest)
```python
from ragdoll.aws_credentials import setup_bedrock_credentials

creds = setup_bedrock_credentials(
    access_key_id="YOUR_AWS_ACCESS_KEY_ID",
    secret_access_key="YOUR_AWS_SECRET_ACCESS_KEY",
    region="us-west-2"
)
```

#### Method B: AWS CLI Profile (Recommended)
Create `~/.aws/credentials`:
```ini
[bedrock-eval]
aws_access_key_id = YOUR_KEY
aws_secret_access_key = YOUR_SECRET
region = us-west-2
```

Then load:
```python
from ragdoll.aws_credentials import setup_from_profile

creds = setup_from_profile(profile="bedrock-eval")
```

#### Method C: Environment Variables
```bash
export AWS_ACCESS_KEY_ID="YOUR_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET"
export AWS_REGION="us-west-2"
```

Then load:
```python
from ragdoll.aws_credentials import load_credentials_from_env

creds = load_credentials_from_env()
```

### 3. Run Evaluation with Bedrock
```bash
uv run ragdoll umbrela judge \
  --input-file examples/umbrela.requests.jsonl \
  --output-file results/umbrela.judgments.jsonl \
  --provider bedrock \
  --model "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
```

## 📚 Key Files

| File | Purpose |
|------|---------|
| `src/ragdoll/aws_credentials.py` | Core credential management |
| `examples/setup_bedrock_credentials.py` | Working examples for all methods |
| `docs/aws_bedrock_setup.md` | Complete documentation |
| `tests/test_aws_credentials.py` | Unit tests |
| `scripts/validate_aws_credentials.py` | Validate setup |

## 🔧 Usage Examples

### Import Directly
```python
from ragdoll import setup_bedrock_credentials, setup_from_profile
```

### Complete Workflow
```python
from pathlib import Path
from ragdoll.aws_credentials import setup_from_profile
from ragdoll.config import RunConfig

# 1. Load credentials
creds = setup_from_profile(profile="bedrock-eval")
assert creds, "Failed to load credentials"

# 2. Create config
config = RunConfig(
    input_file=Path("examples/umbrela.requests.jsonl"),
    output_file=Path("results/umbrela.judgments.jsonl"),
    provider="bedrock",
    model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    raw_events_dir=Path("results/umbrela.raw-events"),
)

print(f"✓ Ready to run evaluation with {config.model}")
```

## 🔐 Security Best Practices

1. **Never hardcode credentials** - Use profiles or environment variables
2. **Use AWS CLI profiles** - Store credentials in `~/.aws/credentials`
3. **Rotate credentials regularly** - Change keys on a schedule
4. **Use temporary credentials** - Enable session tokens for short-lived access
5. **Restrict IAM permissions** - Only allow Bedrock InvokeModel API calls

### Minimal IAM Policy
```json
{
    "Effect": "Allow",
    "Action": ["bedrock:InvokeModel"],
    "Resource": "arn:aws:bedrock:*:ACCOUNT:foundation-model/*"
}
```

## 🤖 Available Bedrock Models

```python
# Claude 3.5 Sonnet (Recommended)
"bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"

# Claude 3 Opus  
"bedrock/anthropic.claude-3-opus-20250219-v1:0"

# Claude 3 Haiku
"bedrock/anthropic.claude-3-haiku-20250307-v1:0"

# Llama 2
"bedrock/meta.llama2-13b-chat-v1"
```

## ✅ Verify Setup

Run the validation script:
```bash
python scripts/validate_aws_credentials.py
```

Or run tests:
```bash
uv run pytest tests/test_aws_credentials.py -v
```

## 📖 Full Documentation

See [`docs/aws_bedrock_setup.md`](../docs/aws_bedrock_setup.md) for:
- Complete API reference
- Setup instructions for all methods
- Troubleshooting guide
- STS temporary credentials
- Environment configuration

## 🆘 Troubleshooting

### Credentials not loading
```python
from ragdoll.aws_credentials import load_credentials_from_env

if load_credentials_from_env() is None:
    print("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
```

### Profile not found
```python
from ragdoll.aws_credentials import setup_from_profile

if setup_from_profile("my-profile") is None:
    print("Profile not found in ~/.aws/credentials")
```

### Bedrock API errors
- Verify region supports Bedrock
- Check IAM permissions
- Confirm model ID is correct
- Ensure model isn't in limited access

## 💡 Next Steps

1. ✅ Install dependencies: `uv sync`
2. ✅ Set up credentials using your preferred method
3. ✅ Verify with: `python scripts/validate_aws_credentials.py`
4. ✅ Run evaluation: `uv run ragdoll umbrela judge --provider bedrock ...`

## 📝 Examples

All methods are demonstrated in:
- `examples/setup_bedrock_credentials.py` - Complete working examples
- `docs/aws_bedrock_setup.md` - Detailed documentation with examples

## 🎯 Common Commands

```bash
# Validate installation
python scripts/validate_aws_credentials.py

# View quick start guide
python scripts/aws_credentials_quickstart.py

# Run tests
uv run pytest tests/test_aws_credentials.py -v

# Run full RAGDoll evaluation
uv run ragdoll umbrela judge \
  --input-file examples/umbrela.requests.jsonl \
  --output-file results/umbrela.judgments.jsonl \
  --provider bedrock \
  --model "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0" \
  --raw-events-dir results/umbrela.raw-events
```

---

**Questions?** Check [`docs/aws_bedrock_setup.md`](../docs/aws_bedrock_setup.md) for comprehensive documentation.
