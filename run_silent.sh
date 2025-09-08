#!/bin/bash
exec "$@" 2> >(grep -v -E "(Schema error|^[[:space:]]*$)" >&2)