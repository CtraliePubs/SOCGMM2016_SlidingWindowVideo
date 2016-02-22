#!/bin/bash

for i in `seq 576 666`;
do
    j=`expr $i - 575`
    mv frames/$i.png frames/$j.png
done
