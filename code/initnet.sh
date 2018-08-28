#!/bin/bash

a=`date +%m%d`
mkdir ../models/$a
mkdir ../models/$a/$1
cp net_master.py ../models/$a/$1/net.py
# vim ../models/$a/$1/net.py
cp run.sh ../models/$a/$1/run.sh

cd ../models/$a/$1

