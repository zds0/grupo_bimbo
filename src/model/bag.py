from __future__ import division
from collections import defaultdict
from glob import glob
import math
import os
import time


def kaggle_bag(glob_files, loc_outfile, method="average", weights="uniform"):
    if method == "average":
        scores = defaultdict(float)
    with open(loc_outfile, "wb") as outfile:
        for i, glob_file in enumerate(glob(glob_files)):
            print "parsing:", glob_file
            # sort glob_file by first column, ignoring the first line
            lines = open(glob_file).readlines()
            lines = [lines[0]] + sorted(lines[1:])
            for e, line in enumerate(lines):
                if i == 0 and e == 0:
                    outfile.write(line)
                if e > 0:
                    row = line.strip().split(",")
                    if scores[(e, row[0])] == 0:
                        scores[(e, row[0])] = 1
                    scores[(e, row[0])] *= float(row[1])
        for j, k in sorted(scores):
            outfile.write("%s, %f\n" %
                          (k, math.pow(max(scores[(j, k)], 0), 1 / (i + 1))))
        print "wrote final submission to %s" % loc_outfile


def ensemble_models():
    print 'Ensembling model predictions...'
    submission_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                                  'models', 'submissions')
    glob_files = submission_dir + '/*submission.csv'
    ts = time.strftime("%c").replace(' ', '-')
    loc_outfile = submission_dir + '/ensembled_' + ts + '.csv'

    kaggle_bag(glob_files, loc_outfile)


if __name__ == '__main__':
    ensemble_models()
