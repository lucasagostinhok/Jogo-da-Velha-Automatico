import tkinter as tk
from tkinter import messagebox
import random
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Variáveis globais
current_player = "X"
board = [" " for _ in range(9)]
buttons = []
game_data = []
model = None  #Armazena o modelo treinado
games_played = 0  #Contador de jogos
max_games = 10  #Número de jogos antes de treinar o modelo


# Função para exibir a mensagem do vencedor
def check_winner():
    global current_player
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                      (0, 3, 6), (1, 4, 7), (2, 5, 8),
                      (0, 4, 8), (2, 4, 6)]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] != " ":
            return current_player
    if " " not in board:
        return "Tie"
    return None


# Função para reiniciar o jogo
def reset_game():
    global current_player, board
    current_player = "X"
    board = [" " for _ in range(9)]
    for button in buttons:
        button.config(text=" ", state=tk.NORMAL)


# Função para registrar dados de cada jogada
def record_data(move):
    global current_player
    game_data.append({"board": board[:], "move": move, "player": current_player})


# Função para jogar automaticamente
def auto_move():
    if model is not None and len(game_data) > 10:
        # Prever o melhor movimento usando o modelo
        board_state = [1 if v == "X" else -1 if v == "O" else 0 for v in board]
        predicted_move = model.predict([board_state])[0]
        return predicted_move
    else:
        # Escolher movimento aleatório se o modelo não estiver treinado
        available_moves = [i for i in range(9) if board[i] == " "]
        return random.choice(available_moves)


# Função de clique nos botões
def play_turn():
    global current_player, model, games_played
    if check_winner():
        return

    # Computador joga
    move = auto_move()
    board[move] = current_player
    buttons[move].config(text=current_player, state=tk.DISABLED)

    # Registrar os dados
    record_data(move)

    # Verificar se há vencedor
    winner = check_winner()
    if winner:
        if winner == "Tie":
            print("Empate!")
        else:
            print(f"Jogador {winner} venceu!")
        games_played += 1
        print(f"Jogo {games_played} concluído.")
        reset_game()

        if games_played >= max_games:
            train_model()
            print("Treinamento realizado após 10 jogos.")
        else:
            # Continuar jogando após reset
            window.after(1000, play_turn)
        return

    # Alternar entre os jogadores
    current_player = "O" if current_player == "X" else "X"

    # Jogar novamente de forma automática
    window.after(1000, play_turn)


# Função para treinar o modelo de árvore de decisão
def train_model():
    global model
    if len(game_data) < 10:
        print("Jogos insuficientes para treinar o modelo.")
        return

    # Transformar os dados coletados em formato utilizável
    df = pd.DataFrame(game_data)
    X = df['board'].apply(lambda x: [1 if v == "X" else -1 if v == "O" else 0 for v in x]).tolist()
    y = df['move']

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Criar e treinar o modelo
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Prever nos dados de teste
    y_pred = clf.predict(X_test)

    # Avaliar a acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

    # Ajuste de Hiperparâmetros
    params = {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 10, 20]}
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=2)
    grid_search.fit(X_train, y_train)

    print("Melhores hiperparâmetros:", grid_search.best_params_)

    # Atualizar o modelo treinado
    model = grid_search.best_estimator_

    # Avaliar o modelo com os melhores hiperparâmetros
    y_pred_best = model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred_best)
    print(f"Acurácia após ajuste de hiperparâmetros: {best_accuracy * 100:.2f}%")


# Função para exibir a janela do jogo
def create_window():
    global buttons, window
    window = tk.Tk()
    window.title("Jogo da Velha (Automático)")

    # Criar botões
    for i in range(9):
        button = tk.Button(window, text=" ", font=("Arial", 24), width=5, height=2,
                           state=tk.DISABLED)
        button.grid(row=i // 3, column=i % 3)
        buttons.append(button)

    # Iniciar o primeiro jogo
    window.after(1000, play_turn)

    # Loop principal da janela
    window.mainloop()


if __name__ == "__main__":
    create_window()
