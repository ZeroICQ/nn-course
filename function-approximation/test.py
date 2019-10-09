import matplotlib.pyplot as plt
import numpy as np

def source_function(x):
    return x-1

def main():
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    plt.plot(y, x)
    plt.show()


if __name__ == '__main__':
    main()