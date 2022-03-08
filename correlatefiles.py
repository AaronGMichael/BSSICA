# returns correlation between lists
import numpy as np

def correlation(listx, listy):
    if len(listx) == 0 or len(listy) == 0:
        # Error checking in main program should prevent us from ever being
        # able to get here.
        raise Exception('Empty lists cannot be correlated.')
    if len(listx) > len(listy):
        listx = listx[:len(listy)]
    elif len(listx) < len(listy):
        listy = listy[:len(listx)]

    covariance = np.cov(listx, listy)
    # for i in range(len(listx)):
    #     covariance += 32 - bin(listx[i] ^ listy[i]).count("1")
    covariance = covariance/ 1.66702963e+15

    return covariance / 32