import pygame
import pygame.gfxdraw
import sys
import math
import numpy as np
import threading

class EvolutionGraph:
    """
    Classe para visualizar a evolução das perdas ao longo do treinamento do MLP
    usando Pygame como biblioteca gráfica.
    """
    
    def __init__(self, width=900, height=600, max_score=2.0, title="Evolução do Treinamento"):
        """Inicializa o gráfico de evolução"""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont('Arial', 15)
        self.font_small = pygame.font.SysFont('Arial', 10)
        self.evolution = []  # Lista para armazenar perdas ao longo das épocas
        self.running = True
        self.max_score = max_score  # Valor máximo esperado no eixo Y
        self.thread = None
        
    def add_score(self, score):
        """Adiciona um novo valor de perda à evolução"""
        self.evolution.append(score)
        
    def draw(self):
        """Desenha o gráfico com os dados atuais"""
        # Configuração de fundo
        self.screen.fill((150, 150, 150))
        
        # Texto dos eixos
        epoch_text = self.font_large.render("Época", True, (0, 0, 0))
        loss_text = self.font_large.render("Perda", True, (0, 0, 0))
        
        # Posicionamento do texto nos eixos
        self.screen.blit(epoch_text, (self.width//2, self.height-10))
        
        # Texto do eixo Y (rotacionado)
        rotated_loss = pygame.Surface((loss_text.get_height(), loss_text.get_width()))
        rotated_loss.fill((150, 150, 150))
        for y in range(loss_text.get_width()):
            for x in range(loss_text.get_height()):
                rotated_loss.set_at((x, loss_text.get_width() - y - 1), 
                                   loss_text.get_at((y, x)))
        self.screen.blit(rotated_loss, (10, self.height//2 - loss_text.get_width()//2))
        
        # Desenhar marcações do eixo X
        x = 50
        y = self.height - 35
        max_epochs = max(50, len(self.evolution))
        xbuff = (self.width - 50) / (max_epochs + 1)
        ybuff = (self.height - 50) / self.max_score
        
        # Desenhar números no eixo X (épocas)
        step = max(1, max_epochs // 10)  # Mostrar no máximo 10 marcações para clareza
        for i in range(0, max_epochs + 1, step):
            text = self.font_small.render(str(i), True, (0, 0, 0))
            x_pos = 50 + i * xbuff
            self.screen.blit(text, (x_pos - text.get_width()//2, y))
        
        # Desenhar números e linhas no eixo Y (perdas)
        x = 35
        y = self.height - 50
        num_divisions = 10
        step_value = self.max_score / num_divisions
        step_pixels = (self.height - 50) / num_divisions
        
        for i in range(num_divisions + 1):
            val = i * step_value
            y_pos = self.height - 50 - i * step_pixels
            text = self.font_small.render(f"{val:.3f}", True, (0, 0, 0))
            self.screen.blit(text, (x - text.get_width(), y_pos - text.get_height()//2))
            pygame.draw.line(self.screen, (0, 0, 0), (50, y_pos), (self.width, y_pos), 1)
        
        # Desenhar a linha de evolução
        if len(self.evolution) > 1:
            for i in range(1, len(self.evolution)):
                prev_loss = min(self.evolution[i-1], self.max_score)  # Limitar ao máximo visível
                curr_loss = min(self.evolution[i], self.max_score)  # Limitar ao máximo visível
                
                start_x = 50 + ((i-1) * xbuff)
                start_y = self.height - 50 - (prev_loss * ybuff)
                end_x = 50 + (i * xbuff)
                end_y = self.height - 50 - (curr_loss * ybuff)
                
                pygame.draw.line(self.screen, (255, 0, 0), (start_x, start_y), (end_x, end_y), 2)
        
        # Desenhar eixos principais
        pygame.draw.line(self.screen, (0, 0, 0), (50, 0), (50, self.height-50), 5)
        pygame.draw.line(self.screen, (0, 0, 0), (50, self.height-50), (self.width, self.height-50), 5)
        
        # Mostrar perda atual se houver dados
        if len(self.evolution) > 0:
            current_loss = self.evolution[-1]
            loss_info = self.font_large.render(f"Perda atual: {current_loss:.6f}", True, (0, 0, 0))
            self.screen.blit(loss_info, (self.width - loss_info.get_width() - 10, 10))
        
        # Atualizar a tela
        pygame.display.flip()
        
    def run_in_thread(self):
        """Executa o gráfico em uma thread separada"""
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True  # Permite que o programa principal termine mesmo se a thread estiver rodando
        self.thread.start()
        
    def run(self):
        """Loop principal para manter a janela aberta e responder a eventos"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            self.draw()
            self.clock.tick(30)  # 30 FPS
        
        pygame.quit()
        
    def exit(self):
        """Fecha o gráfico"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)  # Espera até 1 segundo pela thread terminar
