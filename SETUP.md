# Configuration Setup

## Quick Start

1. Copy the configuration template:
```bash
cp config_vars_template.yml config_vars.yml
```

2. Edit `config_vars.yml` with your settings:
```bash
nano config_vars.yml
# or
vim config_vars.yml
```

3. Update the following fields if you want to use Comet ML for experiment tracking:
   - `comet_ml_api_key`: Get your API key from https://www.comet.com/api/my/settings
   - `comet_ml_workspace`: Your Comet ML workspace name
   - `comet_ml_enable`: Set to `true` to enable tracking

## Important Notes

- **NEVER commit `config_vars.yml` to git** - it contains your personal API keys
- The `config_vars.yml` file is gitignored for security
- Always use `config_vars_template.yml` as the reference for available options
- If you accidentally commit your API key, revoke it immediately and generate a new one

## Configuration Options

See `config_vars_template.yml` for all available configuration options and their descriptions.