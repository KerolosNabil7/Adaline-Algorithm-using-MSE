from Preprocessing import *
from Gui import *
import matplotlib.pyplot as plt


def bias_or_not(bais_bool, x1, x2, w1, w2, bias):
    if bais_bool:
        return bias + w1 * x1 + w2 * x2
    else:
        return w1 * x1 + w2 * x2

def train(x1, x2, t, n, MSE_threshold, bias_bool, w1, w2, bias):
    for i in range(n):
        for j in range(len(x1)):
            y = bias_or_not(bias_bool, x1[j], x2[j], w1, w2, bias)
            error = t[j] - y
            w1 = w1 + eta * error * x1[j]
            w2 = w2 + eta * error * x2[j]
            if bias_bool:
                bias = bias + eta * error

        # Calculate MSE
        MSE = 0.0
        for k in range(len(x1)):
            y = bias_or_not(bias_bool, x1[k], x2[k], w1, w2, bias)
            error = t[k] - y
            MSE = MSE + (error ** 2)
        MSE = (1 / 2) * (MSE / len(x1))

        if MSE <= MSE_threshold:
            break
    if not bias_bool:
        bias = 0
    return w1, w2, bias


def test(x1, x2, t, bias_bool, w1, w2, bias):
    c = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(x1)):
        if bias_bool:
            v = bias + w1 * x1[i] + w2 * x2[i]
        else:
            v = w1 * x1[i] + w2 * x2[i]
        if v >= 0:
            y = 1
        else:
            y = -1

        error = t[i] - y
        if error != 0:
            c = c + 1

        # Confusion matrix
        if t[i] == 1 and y == 1:
            TP += 1
        if t[i] == -1 and y == 1:
            FP += 1
        if t[i] == 1 and y == -1:
            FN += 1
        if t[i] == -1 and y == -1:
            TN += 1
    error = (c / len(x1)) * 100
    acc = 100 - error
    print("Accuracy: " + str(acc))
    print('\n                         Predicted value')
    print('                             P    |    N    ')
    print('')
    print('                      P      ' + str(TP) + '    |   ' + str(FN))
    print('    Actual value -----------------------')
    print('                      N      ' + str(FP) + '    |   ' + str(TN))


# Reading the file
data = pd.read_csv("penguins.csv")

# Preprocessing filling the missing values randomly
data = fill_missing_value(data)

# Manual label Encoder
data['gender'] = lbl_encoder(data['gender'])

# Scaling
data['bill_length_mm'] = scaling(data['bill_length_mm'], 0, 1)
data['bill_depth_mm'] = scaling(data['bill_depth_mm'], 0, 1)
data['flipper_length_mm'] = scaling(data['flipper_length_mm'], 0, 1)
data['body_mass_g'] = scaling(data['body_mass_g'], 0, 1)

c1 = data[:50]
c2 = data[50:100]
c3 = data[100:]

# Shuffled Data
c1 = c1.sample(frac=1, random_state=1).reset_index()
c2 = c2.sample(frac=1, random_state=1).reset_index()
c3 = c3.sample(frac=1, random_state=1).reset_index()

# Inputs
feature1 = selected1.get()
feature2 = selected2.get()
class1 = selected3.get()
class2 = selected4.get()
eta = eta.get()
epochs = epochs.get()
bias_bool = bias.get()
MSE_threshold = MSE_threshold.get()

# Train-Test split
training_dataset = None
testing_dataset = None
all_data = None
if (class1 == 'Adelie' and class2 == 'Gentoo') or (class1 == 'Gentoo' and class2 == 'Adelie'):
    all_data = pd.concat([c1, c2])
    training_dataset = pd.concat([c1.iloc[:30, :], c2.iloc[:30, :]])
    testing_dataset = pd.concat([c1.iloc[30:, :], c2.iloc[30:, :]])
elif (class1 == 'Adelie' and class2 == 'Chinstrap') or (class1 == 'Chinstrap' and class2 == 'Adelie'):
    all_data = pd.concat([c1, c3])
    training_dataset = pd.concat([c1.iloc[:30, :], c3.iloc[:30, :]])
    testing_dataset = pd.concat([c1.iloc[30:, :], c3.iloc[30:, :]])
elif (class1 == 'Gentoo' and class2 == 'Chinstrap') or (class1 == 'Chinstrap' and class2 == 'Gentoo'):
    all_data = pd.concat([c2, c3])
    training_dataset = pd.concat([c2.iloc[:30, :], c3.iloc[:30, :]])
    testing_dataset = pd.concat([c2.iloc[30:, :], c3.iloc[30:, :]])

# Shuffling
training_dataset = training_dataset.sample(frac=1, random_state=1).reset_index()
testing_dataset = testing_dataset.sample(frac=1, random_state=1).reset_index()

# Training Dataset
x1_train = np.array(training_dataset[feature1])
x2_train = np.array(training_dataset[feature2])
t_train = np.array(species_encoder(class1, class2, training_dataset['species']))

# Testing Dataset
x1_test = np.array(testing_dataset[feature1])
x2_test = np.array(testing_dataset[feature2])
t_test = np.array(species_encoder(class1, class2, testing_dataset['species']))

# Hyper Parameters
w1 = np.random.random()
w2 = np.random.random()
b = np.random.random()

# Train
w1, w2, b = train(x1_train, x2_train, t_train, epochs, MSE_threshold, bias_bool, w1, w2, b)

# Draw the graph
x1 = np.array((all_data[feature1]))
x2 = np.array((all_data[feature2]))
t = np.array(species_encoder(class1, class2, all_data['species']))

plt.figure()
plt.scatter(x1[t == 1], x2[t == 1], s=3, c='red')
plt.xlabel(feature1, fontsize=15)
plt.scatter(x1[t == -1], x2[t == -1], s=3, c='green')
plt.ylabel(feature2, fontsize=15)

# First point
X1 = np.min(x1)
Y1 = -(w1 * X1 + b) / w2
# Second point
X2 = np.max(x1)
Y2 = -(w1 * X2 + b) / w2

p1 = [X1, X2]
p2 = [Y1, Y2]

# Decision Boundary
plt.plot(p1, p2)
plt.show()

# Test
test(x1_test, x2_test, t_test, bias_bool, w1, w2, b)
