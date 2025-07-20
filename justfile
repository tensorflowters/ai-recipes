set positional-arguments
set dotenv-load
set fallback

[private]
@default:
  just --list

[unix]
[group("uv")]
[doc("Create a new category of packages")]
@create_package_category category :
  uv init --bare --no-workspace {{category}}

[unix]
[group("uv")]
[doc("Create a new package")]
@create_package category package :
  uv init --package --lib {{category}}/{{package}}

[unix]
[group("project")]
[doc("Setup the development environment for a package")]
@setup package='ambiguity-clarity':
    #!/usr/bin/env bash
    set -e
    echo "Setting up development environment for {{package}}..."
    cd prompt/{{package}}
    uv venv --seed
    uv pip install -e .[dev]
    echo "✅ Done! Activate the virtual environment with: cd prompt/{{package}} && source .venv/bin/activate"

[unix]
[group("project")]
[doc("Run the tests")]
@test package='ambiguity-clarity':
    cd prompt/{{package}}
    pytest -m integration -s