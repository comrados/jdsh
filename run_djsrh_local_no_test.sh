echo "Loading venv..."
source /home/george/Code/venvs/venv/bin/activate

cd /home/george/Code/jdsh/

if [ $# -ne 3 ]; then
    echo "Illegal number of parameters, 3 arguments (model's tag, dataset, data amount) required"
else

  echo "DJSRH 8"
  #/home/george/Code/venvs/venv/bin/python /home/george/Code/jdsh/main.py --bit 8 --tag $1 --dataset $2 --data-amount $3 --model DJSRH

  echo "DJSRH 16"
  /home/george/Code/venvs/venv/bin/python /home/george/Code/jdsh/main.py --bit 16 --tag $1 --dataset $2 --data-amount $3 --model DJSRH

  echo "DJSRH 32"
  /home/george/Code/venvs/venv/bin/python /home/george/Code/jdsh/main.py --bit 32 --tag $1 --dataset $2 --data-amount $3 --model DJSRH

  echo "DJSRH 64"
  /home/george/Code/venvs/venv/bin/python /home/george/Code/jdsh/main.py --bit 64 --tag $1 --dataset $2 --data-amount $3 --model DJSRH

  echo "DJSRH 128"
  /home/george/Code/venvs/venv/bin/python /home/george/Code/jdsh/main.py --bit 128 --tag $1 --dataset $2 --data-amount $3 --model DJSRH

  echo "DJSRH 256"
  #/home/george/Code/venvs/venv/bin/python /home/george/Code/jdsh/main.py --bit 256 --tag $1 --dataset $2 --data-amount $3 --model DJSRH

fi