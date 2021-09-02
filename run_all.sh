if [ $# -ne 1 ]; then
    echo "Illegal number of parameters, 1 argument (data amount: 'normal' or 'aug') required"
else

  ./run_local_no_test.sh default RSICD $1
  ./run_local_no_test.sh default UCM $1

fi