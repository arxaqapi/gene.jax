#!/bin/bash

count_lines() {
    echo `cat $1 | wc -l`
}

clean_logs() {
    for file in log/*.log; do
        if [[ $(count_lines $file) -eq 0 ]]; then
            echo "Deleting: $file "
            rm $file
        fi
    done
}

clean_logs