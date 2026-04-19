# Token Files

This project expects local token files in this directory.

Required files:

- `.hf_token`: Hugging Face access token
- `.gh_token`: GitHub access token
- `.openrouter_token`: OpenRouter API token

Example:

```bash
mkdir -p keys
echo "hf_..." > keys/.hf_token
echo "ghp_..." > keys/.gh_token
echo "sk-or-..." > keys/.openrouter_token
```

These secret files are ignored by git. Commit this `README.md`, not the tokens.
