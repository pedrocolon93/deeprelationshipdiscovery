#!/usr/bin/env bash
echo "Starting shell script.."
echo $1
echo $2
echo $3
echo "param done"
cd "$2"
CUDA_VISIBLE_DEVICES=1 "$1" code/attract-repel.py "$3"
