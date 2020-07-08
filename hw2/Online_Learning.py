import numpy as np
import math

def likelihood(line_data):
    m = line_data.count('1')
    N = len(line_data)
    theta = m/N
    choose = math.factorial(N)/(math.factorial(N-m)*math.factorial(m))
    likelihood_value = choose*math.pow(theta,m)*math.pow((1-theta),(N-m))
    return m, N, likelihood_value

if __name__ == '__main__':
    a = int(input("a= "))
    b = int(input("b= "))
    f = open('testfile.txt')
    line = f.readline()
    cnt = 0
    while line :
        line=line.strip('\n')
        cnt+=1
        m, N, likelihood_value = likelihood(line)
        print('case ',cnt,': ',line)
        print('Likelihood: ', likelihood_value)
        print('Beta prior:     a = ', a, ' b = ', b)
        a = m+a
        b = N-m+b
        print('Beta posterior: a = ', a, ' b = ', b)
        print('')
        line = f.readline()
    f.close()