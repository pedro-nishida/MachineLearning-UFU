import tkinter as tk

#Entrada
A = [-1, -1, 1, 1]
B = [-1, 1, -1, 1]

#Funções Lógicas
    #NULL
Null = [-1 for i in range(4)]
    #Identidade
Id = [1 for i in range(4)]
    #Transfer
TA = A
TB = B
    #NOT
NA = [-x for x in A]
NB = [-x for x in B]
    #ANDs
AND   = [1 if A[i]==1 and B[i]==1 else -1 for i in range(4)]
NAND = [-x for x in AND]
    #ORs
OR   = [1 if A[i]==1 or B[i]==1 else -1 for i in range(4)]
NOR = [-x for x in OR]
    #XORs
XOR   = [1 if A[i]!=B[i] else -1 for i in range(4)]
XNOR  = [-x for x in XOR]
    #Implica
AimpB = [-1 if A[i]==1 & A[i]!=B[i] else 1 for i in range(4)]
BimpA = [-1 if B[i]==1 & A[i]!=B[i] else 1 for i in range(4)]
    #Inibição
AinB = [1 if A[i]==1 and B[i]==-1 else -1 for i in range(4)]
BinA = [1 if B[i]==1 and A[i]==-1 else -1 for i in range(4)] 


Funcoes = { 'Null':Null, 
	    'Id':Id,
	    'TA':TA, 
            'TB':TB, 
            '~A':NA, 
            '~B':NB, 
            'A&B':AND, 
            '~(A&B)':NAND, 
            'A|B':OR, 
            '~(A|B)':NOR,  
            'A^B':XOR, 
            '~(A^B)':XNOR,
            'A->B':AimpB,
            'B->A':BimpA,
            'A&~B':AinB,
	    'B&~A':BinA
        }


# Função para aplicar a Regra de Hebb
def regula_hebb(A, B, R, Limiar):
    w1,w2,b = 0,0,0
    for i in range(4):
        x1,x2,t = A[i],B[i],R[i]
        #Ajuste
        deltaW1 = x1*t
        deltaW2 = x2*t
        deltaB = t
        #Novos valores
        w1 += deltaW1
        w2 += deltaW2
        b += deltaB
    return (w1,w2,b)


def MAIN():
    limiar = 0
    for key in Funcoes:
        R = Funcoes[key]
        #Aplica a regra de Hebb para projetar o neurônio
        w1, w2, b = RegraHebb(A,B,R, limiar)
        #Testa se os resultados batem e se o neurônio funciona
        RH = [1 if (A[i]*w1) + (B[i]*w2) + b >= limiar else -1 
                for i in range(4)] #Resultado de Hebb

        if len([1 for i in range(4) if R[i]!=RH[i]]) != 0:
            print(f"\nProblema com {key}\nParâmetros:{w1=} {w2=} {b=}\n{A=}\n{B=}")
            print(f"Esperado  {R}\nEncontrado{RH}\n")
        else:
            print(f"{key} parâmetros:{w1=} {w2=} {b=}  -OK-")

# Função para atualizar a interface com os resultados
def atualizar_interface():
    # Implemente a lógica para atualizar os elementos da interface
    # com os resultados do neurônio
    # ...

# Crie a janela principal
root = tk.Tk()
root.title("Neurônio Simples")

# Crie os elementos da interface, como rótulos, botões, etc.
# ...

# Defina a função para chamar quando o botão for clicado
def executar_neuronio():
    # Chame as funções lógicas e a Regra de Hebb aqui
    # Atualize a interface com os resultados usando a função atualizar_interface()
    pass

# Crie o botão que executará o neurônio
executar_button = tk.Button(root, text="Executar Neurônio", command=executar_neuronio)
executar_button.pack()

# Inicie o loop principal da interface gráfica
root.mainloop()
