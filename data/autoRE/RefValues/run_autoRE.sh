#!/bin/sh

for i in *.pl
do
	filename=$(echo $i | cut -d "_" -f 2 | cut -d "." -f 1)
	out="result_${filename}.txt"
	err="error_${filename}.txt"
	echo $i
	echo $out
	echo $err
	perl $i 1> $out 2> $err &
done

