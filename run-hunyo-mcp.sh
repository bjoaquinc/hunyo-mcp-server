#!/bin/bash
cd /Users/fatimaarkin/projects/hunyo-notebook-memories-mcp
export HUNYO_DATA_DIR="$PWD/.hunyo"
exec hatch run hunyo-mcp-server "$@" 