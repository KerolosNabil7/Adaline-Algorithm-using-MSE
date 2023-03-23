import tkinter as tk

window = tk.Tk()

window.title('Deep learning model')
window.geometry("700x450")

tk.Label(window, text="Choose Features").place(x=15, y=30)
options = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g']


def valueRemove(selection):
    options.remove(selection)
    tk.OptionMenu(window, selected2, *options).place(x=450, y=30)


selected1 = tk.StringVar()
selected1.set('None')
selected2 = tk.StringVar()
selected2.set('None')

tk.OptionMenu(window, selected1, *options, command=valueRemove).place(x=150, y=30)

tk.Label(window, text="Choose Classes").place(x=15, y=100)
options2 = ['Adelie', 'Gentoo', 'Chinstrap']


def valueRemove2(selection):
    options2.remove(selection)
    tk.OptionMenu(window, selected4, *options2).place(x=450, y=100)


selected3 = tk.StringVar()
selected3.set('None')
selected4 = tk.StringVar()
selected4.set('None')

tk.OptionMenu(window, selected3, *options2, command=valueRemove2).place(x=150, y=100)

# Learning Rate Entry
eta = tk.DoubleVar()
tk.Label(window, text="Enter Learning Rate ").place(x=15, y=175)
eta_entry = tk.Entry(window)
eta_entry.place(x=160, y=175)

# Epochs Entry
epochs = tk.IntVar()
tk.Label(window, text="Enter number of epochs ").place(x=15, y=220)
epochs_entry = tk.Entry(window)
epochs_entry.place(x=160, y=220)

# MSE threshold Entry
MSE_threshold = tk.DoubleVar()
tk.Label(window, text="Enter MSE threshold ").place(x=15, y=265)
MSE_threshold_entry = tk.Entry(window)
MSE_threshold_entry.place(x=160, y=265)

# Bias Checkbox
bias = tk.BooleanVar()


def check_changed():
    print(bias.get())


tk.Checkbutton(window, text='Check for Bias', command=check_changed, variable=bias, onvalue=True, offvalue=False).place(x=15, y=310)


def call_back(e1, e2, e3):
    e1.set(eta_entry.get())
    e2.set(epochs_entry.get())
    e3.set(MSE_threshold_entry.get())
    window.destroy()


tk.Button(window, text='Start', background="red", width=25,
          command=lambda: call_back(eta, epochs, MSE_threshold)).place(x=250, y=400)
window.mainloop()
