import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_loss(theta: float, input, expected) -> float:
    m = len(input)
    predicted = input.dot(theta)
    difference = (predicted - expected)**2
    J = 1 / (2 * m) * np.sum(difference)
    return J

if __name__ == "__main__":
    df = pd.read_csv("D3.csv")

    cols = df.columns
    y = np.array(df['Y'])
    m = len(y)
    cols = cols[:-1]
    total_thetas : list = []
    total_loss : list = []

    x0 = np.ones((m, 1))
    x1 = np.array(df[cols[0]])
    x2 = np.array(df[cols[1]])
    x3 = np.array(df[cols[2]])
    x1 = x1.reshape(m, 1)
    x2 = x2.reshape(m, 1)
    x3 = x3.reshape(m, 1)
    x = np.hstack((x0, x1, x2, x3))

    theta = np.zeros(4)

    j : list = []
    alpha = 0.05
        
    for i in range(1500):
        predictions = x.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * x.transpose().dot(errors)
        theta -= sum_delta
        loss = find_loss(theta, x, y)
        j.append(loss)

    
    total_thetas.append(theta)
    total_loss.append(j[-1])

    print("Thetas:", total_thetas)
    print("Losses:", total_loss)

    plt.plot(j)
    plt.show()
    
    test_1 = [1, 1, 1]
    test_2 = [2, 0, 4]
    test_3 = [3, 2, 1]

    p1 = theta[0] + test_1[0] * theta[1] + test_1[1] * theta[2] + test_1[2] * theta[3]
    p2 = theta[0] + test_2[0] * theta[1] + test_2[1] * theta[2] + test_2[2] * theta[3]
    p3 = theta[0] + test_3[0] * theta[1] + test_3[1] * theta[2] + test_3[2] * theta[3]

    print(p1)
    print(p2)
    print(p3)
    