echo "Loading venv..."
source /home/george/Code/dadh/venv/bin/activate

cd /home/george/Code/jdsh/

if [ $# -ne 2 ]; then
    echo "Illegal number of parameters, 2 arguments (model's tag, dataset) required"
else

  echo "JDSH 8"
  /home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 8 --tag $1 --dataset $2

  echo "JDSH 16"
  /home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 16 --tag $1 --dataset $2

  echo "JDSH 32"
  /home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 32 --tag $1 --dataset $2

  echo "JDSH 64"
  /home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 64 --tag $1 --dataset $2

  echo "JDSH 128"
  /home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 128 --tag $1 --dataset $2

  echo "DJSRH 8"
  /home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 8 --model DJSRH --tag $1 --dataset $2

  echo "DJSRH 16"
  /home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 16 --model DJSRH --tag $1 --dataset $2

  echo "DJSRH 32"
  /home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 32 --model DJSRH --tag $1 --dataset $2

  echo "DJSRH 64"
  /home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 64 --model DJSRH --tag $1 --dataset $2

  echo "DJSRH 128"
  /home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 128 --model DJSRH --tag $1 --dataset $2

fi