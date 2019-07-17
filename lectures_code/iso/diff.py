import os
import sys

def count_lines(filename):
    fp = open(filename, 'r')
    count = 0
    for line in fp:
        count += 1
    fp.close()
    return count

if len(sys.argv) != 3:
    print 'usage: python diff.py correct-text correct-text'
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]
count1 = count_lines(file1)
count2 = count_lines(file2)

if count1 != count2:
    print 'Comparing two files with different lengths'
    sys.exit(1)

count = 0
sum_error = 0.0
sum_err_perc = 0.0

fp1 = open(file1, 'r')
fp2 = open(file2, 'r')

while True:
    line1 = fp1.readline()
    line2 = fp2.readline()
    if line1 == '\n':
        assert line2 == '\n'
        continue
    elif len(line1) == 0:
        assert len(line2) == 0
        break

    tokens1 = line1.split()
    tokens2 = line2.split()
    assert len(tokens1) == len(tokens2)

    ncoordinates = len(tokens1) - 1
    for i in range(ncoordinates):
        c1 = int(tokens1[i])
        c2 = int(tokens2[i])
        assert c1 == c2

    v1 = float(tokens1[ncoordinates])
    v2 = float(tokens2[ncoordinates])

    if v1 != 0.0:
        err = abs(v2 - v1)
        sum_error += err
        sum_err_perc += (err / abs(v1))

    count += 1

fp1.close()
fp2.close()

print 'Mean error = ' + str(sum_error / float(count))
print 'Mean % error = ' + str(sum_err_perc / float(count)) + ' %'
