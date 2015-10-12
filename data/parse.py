#!/usr/bin/env python2

from sys import argv

def avg_one_file(filename):
    txt = open(filename)

    line = txt.readline().split()

    maxdepth = len(line)
    totals = [0.0] * maxdepth
    iters = 0

    while len(line) > 0:
        for i, val in enumerate(line):
            totals[i] += float(val)
        iters += 1

        line = txt.readline().split()

    avgs = [ x / iters for x in totals ]
    print filename + ',',
    print ', '.join([ str(x) for x in avgs ])

if __name__ == '__main__':
    print("file, vertex shader, assemble primitives, geometry shader, scanline, fragment shader")

    for filename in argv[1:]:
        avg_one_file(filename)
