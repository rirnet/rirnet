#!/bin/bash

if [ $# -eq 1 ]
    then
        mkdir ../models/$1
        cp template_net.py ../models/$1/net.py
        cp run.sh ../models/$1/run.sh

        cd ../models/$1

        echo 'Created new directory'
        pwd
        echo 'Caution! You are now in that folder!'
    else
        echo 'No arguments given! Provide model name'
fi
