echo "Loading venv..."
source /home/george/Code/dadh/venv/bin/activate

cd /home/george/Code/jdsh/

echo "JDSH 8"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 8
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 8 --test

echo "JDSH 16"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 16
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 16 --test

echo "JDSH 32"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 32
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 32 --test

echo "JDSH 64"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 64
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 64 --test

echo "JDSH 128"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 128
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 128 --test

echo "DJSRH 8"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 8 --model DJSRH
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 8 --model DJSRH --test

echo "DJSRH 16"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 16 --model DJSRH
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 16 --model DJSRH --test

echo "DJSRH 32"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 32 --model DJSRH
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 32 --model DJSRH --test

echo "DJSRH 64"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 64 --model DJSRH
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 64 --model DJSRH --test

echo "DJSRH 128"
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 128 --model DJSRH
/home/george/Code/dadh/venv/bin/python /home/george/Code/jdsh/main.py --bit 128 --model DJSRH --test
