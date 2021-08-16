#!/bin/bash
set -e

function echo_command()
{
  echo "+++" "$@"
  "$@"
}


for i in {1..100}
do
    echo_command rm -Rf ~/.pocl
    echo_command rm -Rf ~/$XDG_CACHE_HOME/pocl
    echo_command rm -Rf ~/$XDG_CACHE_HOME/pytools
    echo_command rm -Rf ~/$XDG_CACHE_HOME/pyopencl/
    echo_command python main.py
done
