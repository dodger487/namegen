import random

import pandas as pd
import torch
import torch.nn as nn


# mps_device = torch.device("mps")
mps_device = torch.device("cpu")
device = mps_device

# Extra layer
class RNN(nn.Module):
    def __init__(self, n_years, n_genders, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_years + n_genders + input_size + hidden_size, hidden_size)
        self.relu_i2h = nn.LeakyReLU()
        self.i2o = nn.Linear(n_years + n_genders + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.relu_o2o = nn.LeakyReLU()
        self.o2o_2 = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, year, gender, input, hidden):
        input_combined = torch.cat((year, gender, input, hidden), 1)
        hidden = self.i2h(input_combined)
        hidden = self.relu_i2h(hidden)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.relu_o2o(output)
        output = self.o2o_2(output)
        output = self.dropout(output)
        # output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=mps_device)

# Original
class RNN2(nn.Module):
    def __init__(self, n_years, n_genders, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_years + n_genders + input_size + hidden_size, hidden_size)
        self.relu_i2h = nn.LeakyReLU()
        self.i2o = nn.Linear(n_years + n_genders + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, year, gender, input, hidden):
        input_combined = torch.cat((year, gender, input, hidden), 1)
        hidden = self.i2h(input_combined)
        hidden = self.relu_i2h(hidden)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        # output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=mps_device)

# class RNN3(nn.Module):
#     def __init__(self, n_genders, input_size, hidden_size, output_size):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size

#         # +1 for year
#         self.i2h = nn.Linear(1 + n_genders + input_size + hidden_size, hidden_size)
#         self.relu_i2h = nn.LeakyReLU()
#         self.i2o = nn.Linear(1 + n_genders + input_size + hidden_size, output_size)
#         self.o2o = nn.Linear(hidden_size + output_size, output_size)
#         self.dropout = nn.Dropout(0.1)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, year, gender, input, hidden):
#         input_combined = torch.cat((year, gender, input, hidden), 1)
#         hidden = self.i2h(input_combined)
#         hidden = self.relu_i2h(hidden)
#         output = self.i2o(input_combined)
#         output_combined = torch.cat((hidden, output), 1)
#         output = self.o2o(output_combined)
#         output = self.dropout(output)
#         # output = self.softmax(output)
#         return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=mps_device)


df = pd.read_csv("data.csv")

df["clean_name"] = '^' + df.name.str.lower() + '$'

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

all_years = sorted(list(set(df.year)))
num_years = len(all_years)
all_genders = sorted(list(set(df.gender)))
num_genders = len(all_genders)
all_letters = ''.join(sorted(list(set(df.clean_name.str.cat()))))
num_letters = len(all_letters)

def randomExample(df):
    row = df.sample(1)
    return (row.year.values[0], row.gender.values[0], row.clean_name.values[0])

def yearTensor_orig(year):
    tensor = torch.zeros(1, num_years, device=mps_device)
    tensor[0][all_years.index(year)] = 1
    return tensor

def yearTensor(year):
    return torch.tensor([(year - 1950) / 140], device=mps_device).unsqueeze(0)
    tensor = torch.zeros(1, num_years, device=mps_device)
    tensor[0][all_years.index(year)] = 1
    return tensor


def genderTensor(gender):
    tensor = torch.zeros(1, num_genders, device=mps_device)
    tensor[0][gender.index(gender)] = 1
    return tensor

def inputTensor(name):
    tensor = torch.zeros(len(name)-1, 1, num_letters, device=mps_device)
    for li, letter in enumerate(name[:-1]):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    # letter_indexes.append(num_letters - 1)  # EOS.
    return torch.tensor(letter_indexes, device=mps_device)

def randomTrainingExample(df):
    year, gender, name = randomExample(df)
    year_tensor = yearTensor(year)
    gender_tensor = genderTensor(gender)
    input_tensor = inputTensor(name)
    target_tensor = targetTensor(name)
    return year_tensor, gender_tensor, input_tensor, target_tensor


# Convert entire DataFrame to tensors upfront
def df_to_tensors(df):
    # Initialize tensors for years and genders
    num_rows = len(df)
    # year_tensors = torch.zeros(num_rows, num_years)
    # gender_tensors = torch.zeros(num_rows, num_genders)
    year_tensors = {}
    gender_tensors = {}

    # Initialize dictionaries for unique names
    input_dict = {}
    target_dict = {}
    
    # Populate tensors and dictionaries
    for i, row in df.iterrows():
        year, gender, name = row['year'], row['gender'], row['clean_name']
        
        # Year tensor
        if year not in year_tensors:
            year_tensors[year] = yearTensor(year)
        
        # Gender tensor
        if gender not in gender_tensors:
            gender_tensors[gender] = genderTensor(gender)
        # gender_tensors[i][all_genders.index(gender)] = 1
        
        # Populate input and target tensors if the name is unique
        if name not in input_dict:
            input_tensor = torch.zeros(len(name)-1, 1, num_letters)
            target_tensor = torch.zeros(len(name)-1, dtype=torch.long)
            for li, letter in enumerate(name[:-1]):
                input_tensor[li][0][all_letters.find(letter)] = 1
            target_tensor[:] = torch.tensor([all_letters.find(name[li]) for li in range(1, len(name))])
            
            input_dict[name] = input_tensor
            target_dict[name] = target_tensor
            
    return year_tensors, gender_tensors, input_dict, target_dict

# Create tensors from DataFrame
# year_tensors, gender_tensors, input_tensors, target_tensors = df_to_tensors(df)

# Function to get a random training example
# def random_training_example():
#     i = random.randint(0, len(df) - 1)
#     return year_tensors[i], gender_tensors[i], input_tensors[i], target_tensors[i]

# Test random_training_example
# random_training_example()

def df_to_tensors_dict(df):
    # Initialize tensors for years and genders
    num_rows = len(df)
    # year_tensors = torch.zeros(num_rows, num_years)
    # gender_tensors = torch.zeros(num_rows, num_genders)
    year_tensors = {year: yearTensor(year) for year in all_years}
    gender_tensors = {gender: genderTensor(gender) for gender in all_genders}
    unique_names = list(set(df.clean_name))
    input_tensors = {name: inputTensor(name) for name in unique_names}
    target_tensors = {name: targetTensor(name).unsqueeze(-1) for name in unique_names}
    return year_tensors, gender_tensors, input_tensors, target_tensors


year_tensors, gender_tensors, input_tensors, target_tensors = df_to_tensors_dict(df)

def get_row_tensors(row):
    return (
        year_tensors[row["year"]],
        gender_tensors[row["gender"]],
        input_tensors[row["clean_name"]],
        target_tensors[row["clean_name"]],
        row["count"],
    )

def convert_dict_to_device(d, device):
    return {k: v.to(device) for k, v in d.items()}

year_tensors = convert_dict_to_device(year_tensors, device)
gender_tensors = convert_dict_to_device(gender_tensors, device)
input_tensors = convert_dict_to_device(input_tensors, device)
target_tensors = convert_dict_to_device(target_tensors, device)

def random_training_example_dict(df):
    i = random.randint(0, len(df) - 1)
    return get_row_tensors(df.iloc[i])



criterion = nn.CrossEntropyLoss()

# learning_rate = 0.0005

def train(rnn, optimizer, year_tensor, gender_tensor, input_tensor, target_tensor, count):
    # target_tensor = target_tensor.unsqueeze(-1)
    hidden = rnn.initHidden()

    # rnn.zero_grad()

    loss = 0

    for i in range(input_tensor.size(0)):
        output, hidden = rnn(year_tensor, gender_tensor, input_tensor[i], hidden)
        l = criterion(output, target_tensor[i])
        loss += l * math.log(count)
    
    # zeroing gradients after each iteration    
    optimizer.zero_grad()

    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()

    # gradient clipping
    # torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1)

    # updateing the parameters after each iteration
    optimizer.step()

    # for p in rnn.parameters():
    #     p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_tensor.size(0)

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

n_iters = 100000
print_every = 1000
plot_every = 500
all_losses = []
total_loss = 0  # Reset every plot_every iters

num_years = 1
rnn = RNN(num_years, num_genders, num_letters, 128, num_letters)
rnn.to(device)

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

start = time.time()

# data = [randomTrainingExample(df) for _ in range(n_iters)]

for iter in range(1, n_iters + 1):
    # print(iter)
    # output, loss = train(rnn, optimizer, *data[iter - 1])
    # output, loss = train(rnn, optimizer, *randomTrainingExample(df))
    example = random_training_example_dict(df)
    _, loss = train(rnn, optimizer, *example)
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss, total_loss))
        with torch.no_grad():
            print(f"\ti2h norm: {rnn.i2h.weight.grad.norm()}")
            print(f"\ti2o norm: {rnn.i2o.weight.grad.norm()}")
            print(f"\to2o norm: {rnn.i2h.weight.grad.norm()}")

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0



import numpy as np
def generate_name(model, year, gender, start_str=""):
    model.eval()
    with torch.no_grad():
        output, hidden = rnn(
            yearTensor(year), 
            genderTensor(gender), 
            inputTensor("^_")[0],
            model.initHidden()
        )
        # print(hidden)
        p = np.array(torch.softmax(output, 1).squeeze())
        # print( np.array(all_letters), torch.softmax(output, 1))
        # return torch.softmax(output, 1)
        next_letter = np.random.choice(list(all_letters), p=p)
        # topv, topi = output.topk(1)
        # next_letter = all_letters[topi]
        print(next_letter, end="")

        for i in range(100):
            output, hidden = rnn(
                yearTensor(year), 
                genderTensor(gender), 
                inputTensor(next_letter + "_")[0],
                hidden
            )
            p = np.array(torch.softmax(output, 1).squeeze())
            # print( np.array(all_letters), torch.softmax(output, 1))
            # return torch.softmax(output, 1)
            next_letter = np.random.choice(list(all_letters), p=p)
            if next_letter == "$":
                break
            else:
                print(next_letter, end="")
        print()
        print(output, hidden)

        # idx = output.argmax()



# def randomTrainingExample(df):
    # category_tensor = torch.tensor([row.year.values[0] / num_years], dtype=torch.float)
    # input_line_tensor = lineToTensor(row.clean_name.values[0])
    # target_line_tensor = lineToTensor(row.clean_name.values[0])
    # return category_tensor, input_line_tensor, target_line_tensor

# def randomTrainingPair():
#     category = randomChoice(all_categories)
#     line = randomChoice(category_lines[category])
#     return category, line
# def randomTrainingExample():
#     row = df.sample(1)
#     category_tensor = torch.tensor([row.year.values[0] / num_years], dtype=torch.float)
#     input_line_tensor = lineToTensor(row.clean_name.values[0])
#     target_line_tensor = lineToTensor(row.clean_name.values[0])
#     return category_tensor, input_line_tensor, target_line_tensor

# def lineToTensor(line):
#     tensor = torch.zeros(len(line), 1, num_letters)
#     for li, letter in enumerate(line):
#         tensor[li][0][all_letters.find(letter)] = 1
#     return tensor

