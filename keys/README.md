# Token Files

This project expects local token files in this directory.

Always required:

- `.hf_token`: Hugging Face access token
- `.gh_token`: GitHub access token

Required only when running with `--llm`, depending on `--provider`:

- `.openai_token`: OpenAI API key (for `--provider OPENAI`, the default)
- `.openrouter_token`: OpenRouter API key (for `--provider OPENROUTER`)
- (none) `--provider LOCAL` reaches a vLLM server at `http://localhost:8000/v1`
  serving `google/gemma-4-E2B-it` by default.
  Override: `LLM_LOCAL_BASE_URL`, `LLM_LOCAL_MODEL` env vars.

Example:

```bash
mkdir -p keys
echo "hf_..." > keys/.hf_token
echo "ghp_..." > keys/.gh_token
echo "sk-..." > keys/.openai_token        # optional
echo "sk-or-..." > keys/.openrouter_token # optional
```

These secret files are ignored by git. Commit this `README.md`, not the tokens.
