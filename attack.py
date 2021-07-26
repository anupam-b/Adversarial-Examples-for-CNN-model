import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow.keras

import call_model
from fg_attack import fg
from iterative_attack import iterative

data = pd.read_csv("sign_name.csv")
data.sort_values(by="ModelId", ascending=True, inplace=True)
print(" ID -> SIGN NAME\n")
for i, item in data.iterrows():
	print(" "+str(item['ModelId'])+" -> "+str(item['SignName']))
print("\n")

while True:
    try:
        target = int(input("Enter a class ID to attack (default 1): ") or "1")
        if(target<0 or target>42):
        	print("Insert a valid class ID")
        	continue
    except ValueError:
        print("Insert a number")
        continue
    else:
        break

dataframe = pd.read_csv("sign_name.csv")
target = int(target)

print("\nStart of the attack\n")

model = call_model.load_model_weights("models/weights_cnn.hdf5")

print("++++++++++ IN DISTRIBUTION ATTACK ++++++++++\n")

x, y, masks = call_model.load_samples("Test/sample_labeled", "Test/sample_labeled/labels.txt", target)

y_target = np.zeros((len(x))) + target
y_target = tensorflow.keras.utils.to_categorical(y_target, 43)

y_true = np.zeros((len(x))) + y
y_true = tensorflow.keras.utils.to_categorical(y_true, 43)

call_model.printProgressBar(0, 100, prefix = 'Progress FAST GRADIENT TARGET ATTACK:', suffix = 'Complete', length = 50)
x_fg_target = fg(model, x, y_target, masks, True) # FG TARGET ATTACK
call_model.printProgressBar(100, 100, prefix = 'Progress FAST GRADIENT TARGET ATTACK:', suffix = 'Complete', length = 50)
call_model.save_in_distribution_attack(model, "FG", True, target, x, x_fg_target)

print("\n\n")

call_model.printProgressBar(0, 100, prefix = 'Progress FAST GRADIENT UNTARGET ATTACK:', suffix = 'Complete', length = 50)
x_fg_untarget = fg(model, x, y_true, masks, False) # FG UNTARGET ATTACK
call_model.printProgressBar(100, 100, prefix = 'Progress FAST GRADIENT UNTARGET ATTACK:', suffix = 'Complete', length = 50)
call_model.save_in_distribution_attack(model, "FG", False, target, x, x_fg_untarget)

print("\n\n")

call_model.printProgressBar(0, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)
x_it_target = iterative(model, x, y_target, masks, True) # IT TARGET ATTACK
call_model.printProgressBar(100, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)
call_model.save_in_distribution_attack(model, "IT", True, target, x, x_it_target)

print("\n\n")

call_model.printProgressBar(0, 100, prefix = 'Progress ITERATIVE UNTARGET ATTACK:', suffix = 'Complete', length = 50)
x_it_untarget = iterative(model, x, y_true, masks, False) # IT UNTARGET ATTACK
call_model.printProgressBar(100, 100, prefix = 'Progress ITERATIVE UNTARGET ATTACK:', suffix = 'Complete', length = 50)
call_model.save_in_distribution_attack(model, "IT", False, target, x, x_it_untarget)

print("\n\n\n\n")

print("++++++++++ OUT OF DISTRIBUTION ATTACK ++++++++++\n")
print("+++++ LOGO ATTACK +++++\n")

x, masks = call_model.load_out_samples("Test/logo")

y = np.zeros((len(x))) + target
y = tensorflow.keras.utils.to_categorical(y, 43)

call_model.printProgressBar(0, 100, prefix = 'Progress FAST GRADIENT TARGET ATTACK:', suffix = 'Complete', length = 50)
x_fg_target = fg(model, x, y, masks, True) # FG TARGET ATTACK
call_model.printProgressBar(100, 100, prefix = 'Progress FAST GRADIENT TARGET ATTACK:', suffix = 'Complete', length = 50)
call_model.save_out_distribution_attack(model, "FG", target, "LOGO", x, x_fg_target)

print("\n\n")

call_model.printProgressBar(0, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)
x_it_target = iterative(model, x, y, masks, True) # IT TARGET ATTACK
call_model.printProgressBar(100, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)
call_model.save_out_distribution_attack(model, "IT", target, "LOGO", x, x_it_target)

print("\n\n\n\n")

print("++++++++++ OUT OF DISTRIBUTION ATTACK ++++++++++\n")
print("+++++ BLANK SIGNS ATTACK +++++\n")

x, masks = call_model.load_out_samples("Test/blank")

y = np.zeros((len(x))) + target
y = tensorflow.keras.utils.to_categorical(y, 43)

call_model.printProgressBar(0, 100, prefix = 'Progress FAST GRADIENT TARGET ATTACK:', suffix = 'Complete', length = 50)
x_fg_target = fg(model, x, y, masks, True) # FG TARGET ATTACK
call_model.printProgressBar(100, 100, prefix = 'Progress FAST GRADIENT TARGET ATTACK:', suffix = 'Complete', length = 50)
call_model.save_out_distribution_attack(model, "FG", target, "BLANK", x, x_fg_target)

print("\n\n")

call_model.printProgressBar(0, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)
x_it_target = iterative(model, x, y, masks, True) # IT TARGET ATTACK
call_model.printProgressBar(100, 100, prefix = 'Progress ITERATIVE TARGET ATTACK:', suffix = 'Complete', length = 50)
call_model.save_out_distribution_attack(model, "IT", target, "BLANK", x, x_it_target)

print("\n\n")
