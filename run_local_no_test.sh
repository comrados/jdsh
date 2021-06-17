echo "Loading venv..."
source /home/george/Code/dadh/venv/bin/activate

cd /home/george/Code/jdsh/

echo "JDSH 8"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 8

echo "JDSH 16"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 16

echo "JDSH 32"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 32

echo "JDSH 64"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 64

echo "DJSRH 8"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 8 --model DJSRH

echo "DJSRH 16"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 16 --model DJSRH

echo "DJSRH 32"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 32 --model DJSRH

echo "DJSRH 64"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 64 --model DJSRH

