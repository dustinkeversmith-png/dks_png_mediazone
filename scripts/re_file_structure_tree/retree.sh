#!/usr/bin/env bash

START_DIR="$(pwd)"

tree_bash() {
    local dir="${1:-.}"
    local prefix="$2"
    
    local ignore_patterns=()
    
    load_ignore_file() {
        local file="$1"
        if [ -f "$file" ]; then
            while IFS= read -r line || [ -n "$line" ]; do
                # Trim carriage returns (CRLF) and leading/trailing whitespace
                line=$(echo "$line" | tr -d '\r' | xargs 2>/dev/null)
                # Skip empty lines and comments
                [[ -z "$line" || "$line" =~ ^# ]] && continue
                ignore_patterns+=("$line")
            done < "$file"
        fi
    }

    # Load ignore lists
    load_ignore_file "$dir/.ignore"
    if [ "$dir" != "$START_DIR" ]; then
        load_ignore_file "$START_DIR/.ignore"
    fi

    # Process files including hidden ones
    local shopt_state=$(shopt -p dotglob)
    shopt -s dotglob
    local raw_files=("$dir"/*)
    shopt -u dotglob
    $shopt_state 

    local all_files=()
    for file in "${raw_files[@]}"; do
        local name="${file##*/}"
        [[ "$name" == "." || "$name" == ".." ]] && continue
        
        local skip=0
        for pattern in "${ignore_patterns[@]}"; do
            # Standardize pattern by removing trailing slashes for directory matching
            local clean_pattern="${pattern%/}"
            
            # Robust glob matching against the filename
            if [[ "$name" == $clean_pattern ]]; then
                skip=1
                break
            fi
        done
        
        if [ "$skip" -eq 0 ]; then
            all_files+=("$file")
        fi
    done

    local count=${#all_files[@]}
    local i=0

    for file in "${all_files[@]}"; do
        local name="${file##*/}"
        ((i++))
        
        if [ "$i" -eq "$count" ]; then
            echo "${prefix}└── $name"
            if [ -d "$file" ] && [ ! -L "$file" ]; then
                tree_bash "$file" "${prefix}    "
            fi
        else
            echo "${prefix}├── $name"
            if [ -d "$file" ] && [ ! -L "$file" ]; then
                tree_bash "$file" "${prefix}│   "
            fi
        fi
    done
}

tree_bash "${1:-$START_DIR}"