#!/bin/bash

NX=512
NY=512
INPUT=snap.text

OPTIND=1
while getopts "x:y:i:" opt; do
    case "$opt" in
        x)  NX=$OPTARG
            ;;
        y)  NY=$OPTARG
            ;;
        i) INPUT=$OPTARG
            ;;
    esac
done

gnuplot -e "INPUT='$INPUT'" -e "NX=$NX" -e "NY=$NY" iso.gnu
