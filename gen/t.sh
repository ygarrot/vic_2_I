#/bin/sh
plantcv-utils.py tabulate_bayes_classes -i test_rgb.txt -o bayes_classes.tsv
plantcv-train.py naive_bayes_multiclass --file bayes_classes.tsv --outfile naive_bayes_pdfs.txt --plots
