#!/bin/bash
#SBATCH -o /home/users/m/mikriukov/projects/jdsh/out_gpu_short.log
#SBATCH -J jdsh
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu_short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla:1

echo "Loading venv..."
source /home/users/m/mikriukov/venvs/DADH/bin/activate

echo "Loading cuda..."
module load nvidia/cuda/10.1

echo "JDSH 8"
python3 main.py --bit 8
python3 main.py --bit 8 --test

echo "JDSH 16"
python3 main.py --bit 16
python3 main.py --bit 16 --test

echo "JDSH 32"
python3 main.py --bit 32
python3 main.py --bit 32 --test

echo "JDSH 64"
python3 main.py --bit 64
python3 main.py --bit 64 --test

echo "JDSH 128"
python3 main.py --bit 128
python3 main.py --bit 128 --test

echo "DJSRH 8"
python3 main.py --bit 8 --model DJSRH
python3 main.py --bit 8 --model DJSRH --test

echo "DJSRH 16"
python3 main.py --bit 16 --model DJSRH
python3 main.py --bit 16 --model DJSRH --test

echo "DJSRH 32"
python3 main.py --bit 32 --model DJSRH
python3 main.py --bit 32 --model DJSRH --test

echo "DJSRH 64"
python3 main.py --bit 64 --model DJSRH
python3 main.py --bit 64 --model DJSRH --test

echo "DJSRH 128"
python3 main.py --bit 128 --model DJSRH
python3 main.py --bit 128 --model DJSRH --test
