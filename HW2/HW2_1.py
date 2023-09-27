import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def find_loss(theta, input, expected) -> float:
    m = len(input)
    predicted = input.dot(theta)
    difference = (predicted - expected)**2
    J = 1 / (2 * m) * np.sum(difference)
    return J

def normalize(arr, range_min, range_max):
    norm_arr = []
    diff = range_max - range_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + range_min
        norm_arr.append(temp)
    return norm_arr

def validate(test_x, test_y, thetas) -> float:
    loss = 0
    for input, expected in zip(test_x, test_y):
        loss += find_loss(thetas, input, expected)
    return loss

if __name__ == "__main__":
    df = pd.read_csv("Housing.csv")

    df['mainroad'] = df['mainroad'].apply(lambda x: 1 if x == 'yes' else 0)
    df['guestroom'] = df['guestroom'].apply(lambda x: 1 if x == 'yes' else 0)
    df['basement'] = df['basement'].apply(lambda x: 1 if x == 'yes' else 0)
    df['hotwaterheating'] = df['hotwaterheating'].apply(lambda x: 1 if x == 'yes' else 0)
    df['airconditioning'] = df['airconditioning'].apply(lambda x: 1 if x == 'yes' else 0)
    df['prefarea'] = df['prefarea'].apply(lambda x: 1 if x == 'yes' else 0)
    df['furnishingstatus'] = df['furnishingstatus'].apply(lambda x: 2 if x == 'furnished' else 0)

    temp_y = df.pop('price')
    norm = np.linalg.norm(df)
    df = df/norm
    df['price'] = temp_y
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=15)

    y = np.array(train_df['price'])

    test_y = np.array(test_df['price'])

    m = len(y)
    test_m = len(test_y)

    total_thetas : list = []
    total_loss : list = []

    # Part A
    # factors_list : list[str] = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    # iterations = 7500
    # alpha = 0.05

    # Part B
    factors_list : list[str] = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
                                'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
                                'parking', 'prefarea']
    iterations = 100
    alpha = 0.05

    x = np.ones((m, 1))
    test_x = np.ones((test_m, 1))
    
    for factor in factors_list:
        temp = np.array(train_df[factor])
        temp = temp.reshape(m, 1)
        x = np.hstack((x, temp))
        temp = np.array(test_df[factor])
        temp = temp.reshape(test_m, 1)
        test_x = np.hstack((test_x, temp))
    
    # Normalization
    # norm = np.linalg.norm(x)
    # x = x/norm

    # test_norm = np.linalg.norm(test_x)
    # test_x = test_x/test_norm

    # Standardization
    # x = (x - np.mean(x)) / np.std(x)

    # test_x = (test_x - np.mean(test_x)) / np.std(test_x)


    theta = np.zeros(len(factors_list)+1)

    j : list = []
    validation_loss : list[float] = []
        
    for i in tqdm(range(iterations), desc="Processing", unit="iteration"):
        predictions = x.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * x.transpose().dot(errors)
        theta -= sum_delta
        loss = find_loss(theta, x, y)
        j.append(loss)
        v = validate(test_x, test_y, theta)
        validation_loss.append(v)

    total_thetas.append(theta)
    total_loss.append(j[-1])

    print("Thetas:", total_thetas)
    print("Losses:", total_loss)
    print("Validation Loss:", validation_loss[-1])

    fig, ax = plt.subplots()

    ax.plot(j, label='Loss')
    ax2 = ax.twinx()
    ax2.plot(validation_loss, label='Validation Loss', color='orange')
    ax.legend()
    ax.set_ylabel('Error')
    ax.set_xlabel('Iteration')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax2.set_ylim(0, ax2.get_ylim()[1])

    # Combine legends for both lines
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.set_title('Normalized')
    plt.show()

    