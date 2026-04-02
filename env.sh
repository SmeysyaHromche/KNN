#!/bin/sh

ENV_DIR=".venv"
PYTHON_BIN="python3.12"
REQUIREMENTS_FILE="requirements.txt"

show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  -c, --create      Create environment if it does not exist"
    echo "  -r, --recreate    Recreate environment from scratch"
    echo "  -d, --delete      Delete environment"
    echo "  -a, --activate    Activate environment (must be sourced)"
    echo "  -h, --help        Show this help message"
}

safe_return() {
    return "$1" 2>/dev/null || exit "$1"
}

create_env() {
    echo "Creating virtual environment..."
    "$PYTHON_BIN" -m venv "$ENV_DIR" || safe_return 1

    echo "Activating environment..."
    . "$ENV_DIR/bin/activate" || safe_return 1

    echo "Upgrading pip..."
    pip install --upgrade pip || safe_return 1

    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "Installing requirements..."
        pip install -r "$REQUIREMENTS_FILE" || safe_return 1
    else
        echo "No requirements.txt found, skipping install."
    fi

    echo "Environment ready."
}

recreate_env() {
    if [ -d "$ENV_DIR" ]; then
        echo "Removing existing environment..."
        rm -rf "$ENV_DIR" || safe_return 1
    fi
    create_env
}

delete_env() {
    if [ -d "$ENV_DIR" ]; then
        echo "Deleting environment..."
        rm -rf "$ENV_DIR" || safe_return 1
        echo "Environment deleted."
    else
        echo "Environment does not exist."
    fi
}

activate_env() {
    if [ ! -d "$ENV_DIR" ]; then
        echo "Environment does not exist. Create it first."
        safe_return 1
    fi

    echo "Activating environment..."
    . "$ENV_DIR/bin/activate" || safe_return 1
}

if [ $# -eq 0 ]; then
    show_help
    safe_return 1
fi

case "$1" in
    -c|--create)
        if [ -d "$ENV_DIR" ]; then
            echo "Environment already exists. Doing nothing."
        else
            create_env
        fi
        ;;
    -r|--recreate)
        recreate_env
        ;;
    -d|--delete)
        delete_env
        ;;
    -a|--activate)
        activate_env
        ;;
    -h|--help)
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        safe_return 1
        ;;
esac