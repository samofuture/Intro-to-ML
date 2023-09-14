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
    col_names = ['X1', 'X2', 'X3']
    fig, ax = plt.subplots(len(cols))
    total_thetas : list = []
    total_loss : list = []
    for index, column in enumerate(cols):
        x0 = np.ones((m, 1))
        x1 = np.array(df[column])
        x1 = x1.reshape(m, 1)
        x = np.hstack((x0, x1))

        theta = np.zeros(2)

        j : list = []
        alpha = .1
        
        for i in range(1000):
            predictions = x.dot(theta)
            errors = np.subtract(predictions, y)
            sum_delta = (alpha / m) * x.transpose().dot(errors)
            theta -= sum_delta
            loss = find_loss(theta, x, y)
            j.append(loss)

        total_thetas.append(theta)
        total_loss.append(j[-1])

        x_vals = np.linspace(min(x1), max(x1), 100)
        h = x_vals * theta[1] + theta[0]

        ax[index].scatter(x1, y, color='red', s=5)
        ax[index].plot(x_vals, h)
        ax[index].set_title(col_names[index])
    
    print("Thetas:", total_thetas)
    print("Losses:", total_loss)
        
    plt.tight_layout()
    plt.show()