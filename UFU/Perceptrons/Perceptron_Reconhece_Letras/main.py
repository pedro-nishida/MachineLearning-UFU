from neuronio import Neuronio
import tkinter as tk
import pickle
import os

def f(Yin, Limiar):
    if Yin >= Limiar: return 1
    return -1

class PixelGridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Letter Recognizer")

        self.result_text = tk.StringVar()  # Variável para exibir o resultado
        self.result_label = tk.Label(self.root, textvariable=self.result_text)
        self.result_label.pack()

        self.pixel_size = 50
        self.grid_data = [[-1 for _ in range(5)] for _ in range(5)]
        self.Data = [] # Armazena os dados para treinamentos
        self.neurons = {}  # Dicionário para armazenar os neurônios
        self.pick_data()

        self.canvas = tk.Canvas(self.root, width=self.pixel_size*5, height=self.pixel_size*5)
        self.canvas.pack()

        self.draw_grid()

        self.canvas.bind("<Button-1>", self.toggle_pixel)
        self.canvas.bind("<B1-Motion>", self.drag_pixel)

        self.entry = tk.Entry(self.root)
        self.entry.pack()

        self.learn_button = tk.Button(self.root, text="Learn", command=self.learn_model)
        self.learn_button.pack()

        self.clean_button = tk.Button(self.root, text="Clean Data", command=self.clean_data)
        self.clean_button.pack()

        self.print_button = tk.Button(self.root, text="Save Data", command=self.save_data)
        self.print_button.pack()

        self.clear_grid_button = tk.Button(self.root, text="Clear Grid", command=self.clear_grid)
        self.clear_grid_button.pack()

    def draw_grid(self):
        self.canvas.delete("all")
        for row in range(5):
            for col in range(5):
                color = "white" if self.grid_data[row][col] == -1 else "black"
                self.canvas.create_rectangle(
                    col * self.pixel_size,
                    row * self.pixel_size,
                    (col + 1) * self.pixel_size,
                    (row + 1) * self.pixel_size,
                    fill=color
                )

    def toggle_pixel(self, event):
        col = event.x // self.pixel_size
        row = event.y // self.pixel_size
        self.grid_data[row][col] *= -1
        self.draw_grid()
        self.update_result_label()

    def drag_pixel(self, event):
        col = event.x // self.pixel_size
        row = event.y // self.pixel_size
        if 0 <= col < 5 and 0 <= row < 5:
            self.grid_data[row][col] = 1
            self.draw_grid()
            self.update_result_label()

    def calculate_outputs(self):
        current_input = self.get_concatenated_data()
        neuron_outputs = {}  # Dicionário para armazenar as saídas dos neurônios
        for letter, neuron in self.neurons.items():
            neuron_outputs[letter] = neuron.output(current_input)
        return neuron_outputs

    def update_result_label(self):
        outputs = self.calculate_outputs()
        max_letter = max(outputs, key=outputs.get)
        self.result_text.set(f"Resultado: {max_letter}")

    def learn_model(self):
        letterResult = self.entry.get()
        inputData = self.get_concatenated_data()
        self.Data.append((inputData, letterResult))
        print(self.Data[-1])

        # Treinar o neurônio correspondente
        self.train_neurons()

    def train_neurons(self):
        unique_letters = set(item[1] for item in self.Data)
        for letter in unique_letters:
            data_for_neuron = [(input_data, 1 if result == letter else -1) for input_data, result in self.Data]
            neuron = Neuronio(f)  # Crie uma instância do neurônio
            neuron.learnPerceptron(data_for_neuron)
            self.neurons[letter] = neuron

    def get_concatenated_data(self):
        concatenated_data = [self.grid_data[i][j] for i in range(5) for j in range(5)]
        return concatenated_data

    def save_data(self):
        with open("data.bin", "wb") as file:
            pickle.dump(self.Data, file)
        print("Data Saved!")

    def pick_data(self):
        if os.path.exists("data.bin"):
            with open("data.bin", "rb") as file:
                self.Data = pickle.load(file)
            for item in self.Data:
                print(item)
            self.train_neurons()
        else:
            print("No data file found. Starting with empty data.")
            with open("data.bin", "wb") as file:
                pickle.dump(self.Data, file)

    def clean_data(self):
        self.Data = []
        self.neurons = {}
        with open("data.bin", "wb") as file:
            pickle.dump(self.Data, file)
        print("Data Cleared!")

    def clear_grid(self):
        self.grid_data = [[-1 for _ in range(5)] for _ in range(5)]
        self.draw_grid()
        self.update_result_label()

if __name__ == "__main__":
    root = tk.Tk()
    app = PixelGridApp(root)
    root.mainloop()