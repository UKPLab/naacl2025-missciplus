#!/bin/bash

covidfact_url="https://raw.githubusercontent.com/asaakyan/covidfact/main/RTE-covidfact"
covidfact_file_train="train.tsv"
covidfact_file_dev="dev.tsv"
covidfact_file_test="test1.tsv"

healthver_url="https://raw.githubusercontent.com/sarrouti/HealthVer/refs/heads/master/data/"
healthver_file_train="healthver_train.csv"
healthver_file_dev="healthver_dev.csv"
healthver_file_test="healthver_test.csv"

echo "Downloading COVIDFact ..."
mkdir -p ./RTE-covidfact/
curl -L "${covidfact_url}/${covidfact_file_train}" -o "./RTE-covidfact/${covidfact_file_train}"
curl -L "${covidfact_url}/${covidfact_file_dev}" -o "./RTE-covidfact/${covidfact_file_dev}"
curl -L "${covidfact_url}/${covidfact_file_test}" -o "./RTE-covidfact/${covidfact_file_test}"
echo "Done."


echo "Downloading HealthVer ..."
mkdir -p ./healthver/
curl -L "${healthver_url}/${healthver_file_train}" -o "./healthver/${healthver_file_train}"
curl -L "${healthver_url}/${healthver_file_dev}" -o "./healthver/${healthver_file_dev}"
curl -L "${healthver_url}/${healthver_file_test}" -o "./healthver/${healthver_file_test}"
echo "Done."
