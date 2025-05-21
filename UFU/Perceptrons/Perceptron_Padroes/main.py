from neuronio import Neuronio
import tkinter as tk
import pickle
import numpy as np
# from sklearn.decomposition import PCA
from scipy.ndimage import zoom


def f(Yin, Limiar):
    if Yin >= Limiar:
        return 1
    return -1


class PixelGridApp:
    def __init__(self, root):
        self.organizing_data = False

        self.root = root
        self.root.title("Letter Recognizer")

        self.result_text = tk.StringVar()  # Variável para exibir o resultado
        self.result_label = tk.Label(self.root, textvariable=self.result_text)
        self.result_label.pack()

        self.tool = "pencil"  # Opções: "pencil", "eraser"
        self.mouse_clicked = False

        self.clean_button = tk.Button(
            self.root, text="Erase All", command=self.clean_tool)
        self.clean_button.pack()

        self.tool_button = tk.Button(
            self.root, text="pencil", command=self.switch_tool)
        self.tool_button.pack()

        self.pixel_size = 10
        self.width = 60
        self.height = 30

        self.grid_data = [
            [-1 for _ in range(self.width)] for _ in range(self.height)]
        self.Data = []  # Armazena os dados para treinamentos
        self.neurons = {}  # Dicionário para armazenar os neurônios
        if not self.organizing_data:
            self.pick_data()

        self.canvas = tk.Canvas(
            self.root, width=self.width*self.pixel_size, height=self.height*self.pixel_size)
        self.canvas.pack()

        self.draw_grid()

        self.canvas.bind("<Button-1>", self.toggle_pixel)

        self.entry = tk.Entry(self.root)
        self.entry.pack()

        self.learn_button = tk.Button(
            self.root, text="Learn", command=self.learn_model)
        self.learn_button.pack()

        self.clean_button = tk.Button(
            self.root, text="Clean Data", command=self.clean_data)
        self.clean_button.pack()

        self.print_button = tk.Button(
            self.root, text="Save Data", command=self.save_data)
        self.print_button.pack()

        self.canvas.bind("<Button-1>", self.toggle_pixel)
        self.canvas.bind("<B1-Motion>", self.track_mouse_motion)
        self.canvas.bind("<ButtonRelease-1>", self.release_mouse)

    def prepare_data(self):
        self.data_matrix = np.array(self.grid_data)
        self.data_matrix = (self.data_matrix - np.mean(self.data_matrix)) / np.std(self.data_matrix)
        self.data_matrix = zoom(self.data_matrix, (10/30, 20/60))
        return self.data_matrix
        # pca = PCA(n_components=self.num_components)
        # data_reduced = pca.fit_transform(self.data_matrix)
        # return data_reduced

    def draw_grid(self):
        self.canvas.delete("all")
        for row in range(self.height):
            for col in range(self.width):
                color = "white" if self.grid_data[row][col] == -1 else "black"
                self.canvas.create_rectangle(
                    col * self.pixel_size,
                    row * self.pixel_size,
                    (col + 1) * self.pixel_size,
                    (row + 1) * self.pixel_size,
                    fill=color, outline="white"
                )

    def toggle_pixel(self, event):
        col = event.x // self.pixel_size
        row = event.y // self.pixel_size

        if self.tool == "pencil":
            self.grid_data[row][col] = 1
        elif self.tool == "eraser":
            self.grid_data[row][col] = -1

        self.draw_grid()
        self.mouse_clicked = True

    def release_mouse(self, event):
        self.mouse_clicked = False
        if not self.organizing_data:
            self.update_result_label()

    def track_mouse_motion(self, event):
        if self.mouse_clicked:
            self.toggle_pixel(event)

    def switch_tool(self):
        if self.tool == "pencil":
            self.tool_button['text'] = self.tool = "eraser"
        else:
            self.tool_button['text'] = self.tool = "pencil"

    def clean_tool(self):
        self.grid_data = [
            [-1 for _ in range(self.width)] for _ in range(self.height)]
        self.draw_grid()

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

    def train_neurons(self):
        unique_letters = set(item[1] for item in self.Data)
        for letter in unique_letters:
            data_for_neuron = [(input_data, 1 if result == letter else -1)
                               for input_data, result in self.Data]
            neuron = Neuronio(f)  # Crie uma instância do neurônio
            neuron.learnPerceptron(data_for_neuron, alpha=2**-3)
            self.neurons[letter] = neuron

    def learn_model(self):
        letterResult = self.entry.get()
        inputData = self.get_concatenated_data()
        self.Data.append((inputData, letterResult))
        print(len(inputData), letterResult)

        # Treinar o neurônio correspondente
        self.train_neurons()

    def get_concatenated_data(self):
        data_reduced = self.prepare_data()
        data_reduced = [data_reduced[i][j]
                             for i in range(len(data_reduced)) for j in range(len(data_reduced[i]))]
        return data_reduced

    def save_data(self):
        with open("data.bin", "wb") as file:
            pickle.dump(self.Data, file)
        print("Data Saved!")

    def pick_data(self):
        with open("data.bin", "rb") as file:
            self.Data = pickle.load(file)
        for item in self.Data:
            print(len(item[0]), " ", item[1])

        self.train_neurons()

    def clean_data(self):
        if not self.organizing_data:
            self.pick_data()
        else:
            self.Data = []
            self.save_data()


if __name__ == "__main__":
    root = tk.Tk()
    app = PixelGridApp(root)
    root.mainloop()
