echo "Loading venv..."
source /home/george/Code/venvs/venv/bin/activate

cd /home/george/Code/jdsh/

if [ $# -ne 3 ]; then
    echo "Illegal number of parameters, 3 arguments (model's tag, dataset, data amount) required"
else

  echo "JDSH 8"
  #python main.py --bit 8 --tag $1 --dataset $2 --data-amount $3

  echo "JDSH 16"
  python main.py --bit 16 --tag $1 --dataset $2 --data-amount $3

  echo "JDSH 32"
  python main.py --bit 32 --tag $1 --dataset $2 --data-amount $3

  echo "JDSH 64"
  python main.py --bit 64 --tag $1 --dataset $2 --data-amount $3

  echo "JDSH 128"
  python main.py --bit 128 --tag $1 --dataset $2 --data-amount $3

  echo "DJSRH 8"
  #python main.py --bit 8 --model DJSRH --tag $1 --dataset $2 --data-amount $3

  echo "DJSRH 16"
  python main.py --bit 16 --model DJSRH --tag $1 --dataset $2 --data-amount $3

  echo "DJSRH 32"
  python main.py --bit 32 --model DJSRH --tag $1 --dataset $2 --data-amount $3

  echo "DJSRH 64"
  python main.py --bit 64 --model DJSRH --tag $1 --dataset $2 --data-amount $3

  echo "DJSRH 128"
  python main.py --bit 128 --model DJSRH --tag $1 --dataset $2 --data-amount $3

fi