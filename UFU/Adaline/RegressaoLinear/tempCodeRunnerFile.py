base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "basedeobservacoes.txt")

with open(file_path, "r") as file:
    for l in file:
        l = l.strip().split(" ")
        Data.append((l[0], l[-1]))
Data = [(float(d[0]), float(d[1])) for d in Data[1:]]
