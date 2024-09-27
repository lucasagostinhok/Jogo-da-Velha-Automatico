import tkinter as tk
from tkinter import messagebox
import random
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# Variáveis globais
current_player = "X"
board = [" " for _ in range(9)]
buttons = []
game_data = []
model = None  # Armazena o modelo treinado
games_played = 0  # Contador de jogos
train_after_games = 10  # Treinar o modelo após 10 jogos
train_counter = 0  # Contador de treinamentos
game_speed = 500  # Tempo de espera entre jogadas

# verifica se há um vencedor
def check_winner():
    global current_player
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                      (0, 3, 6), (1, 4, 7), (2, 5, 8),
                      (0, 4, 8), (2, 4, 6)]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] != " ":
            return board[condition[0]]
    if " " not in board:
        return "Tie"
    return None

# reinicia o jogo
def reset_game():
    global current_player, board
    current_player = "X"
    board = [" " for _ in range(9)]
    for button in buttons:
        button.config(text=" ", state=tk.NORMAL)

# registra os dados de cada jogada
def record_data(move):
    global current_player
    game_data.append({"board": board[:], "move": move, "player": current_player})

# movimento automático
def auto_move():
    available_moves = [i for i in range(9) if board[i] == " "]
    if model is not None and len(game_data) >= train_after_games:
        # Preve o melhor movimento usando o modelo treinado
        board_state = [1 if v == "X" else -1 if v == "O" else 0 for v in board]
        predicted_move = model.predict([board_state])[0]
        if predicted_move in available_moves:
            return predicted_move
        else:
            return random.choice(available_moves) 
    else:
        # Escolhe movimento aleatório se o modelo não estiver treinado
        return random.choice(available_moves)

def play_turn():
    global current_player, model, games_played

    move = auto_move()
    if board[move] == " ": 
        board[move] = current_player
        buttons[move].config(text=current_player, state=tk.DISABLED)

        # Registra os dados
        record_data(move)

        # Verifica se há um vencedor após o movimento
        winner = check_winner()
        if winner:
            if winner == "Tie":
                print("Empate!")
            else:
                print(f"Jogador {winner} venceu!") 
            games_played += 1
            print(f"Jogo {games_played} concluído.")
            reset_game()

            # Verificar se é hora de treinar o modelo
            if games_played % train_after_games == 0:
                try:
                    train_model()
                except Exception as e:
                    print(f"Erro durante o treinamento: {e}")

            window.after(game_speed, play_turn)
            return

        current_player = "O" if current_player == "X" else "X"

    window.after(game_speed, play_turn)

# Função para treinar o modelo de Gradient Boosting com RandomizedSearchCV
def train_model():
    global model, train_counter

    if len(game_data) < train_after_games:
        print("Jogos insuficientes para treinar o modelo.")
        return

    # Transformar os dados coletados em um formato utilizável
    df = pd.DataFrame(game_data)
    X = df['board'].apply(lambda x: [1 if v == "X" else -1 if v == "O" else 0 for v in x]).tolist()
    y = df['move']

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 1. Treinamento normal (sem ajuste de hiperparâmetros)
    clf_normal = GradientBoostingClassifier()
    clf_normal.fit(X_train, y_train)

    # Acurácia normal
    y_pred_normal = clf_normal.predict(X_test)
    normal_accuracy = accuracy_score(y_test, y_pred_normal)
    print(f"Acurácia normal após {games_played} jogos: {normal_accuracy * 100:.2f}%")

    # 2. Ajuste de hiperparâmetros com RandomizedSearchCV
    params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }

    random_search = RandomizedSearchCV(
        GradientBoostingClassifier(),
        param_distributions=params,
        n_iter=5,  # Limite para testar apenas 5 combinações aleatórias
        cv=3,  # Valida com 3 folds
        random_state=42,
        n_jobs=-1 
    )
    random_search.fit(X_train, y_train)

    # Modelo ajustado
    model = random_search.best_estimator_

    # Acurácia ajustada com os melhores hiperparâmetros
    y_pred_adjusted = model.predict(X_test)
    adjusted_accuracy = accuracy_score(y_test, y_pred_adjusted)
    print(f"Acurácia ajustada após {games_played} jogos: {adjusted_accuracy * 100:.2f}% (Melhores hiperparâmetros: {random_search.best_params_})")

    train_counter += 1

    # Salvamento do modelo
    try:
        filename = "trained_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modelo salvo com sucesso como {filename}")
    except Exception as e:
        print(f"Erro ao salvar o modelo: {e}")

# carregar o modelo salvo
def load_model():
    global model
    try:
        with open('trained_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Modelo carregado com sucesso.")
    except FileNotFoundError:
        print("Nenhum modelo salvo encontrado. O modelo será treinado após suficientes jogos.")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")

# ao fechar a janela, mostra a acurácia normal e ajustada
def on_closing():
    global model
    if model is not None:
        # transformar os dados coletados em formato utilizável
        df = pd.DataFrame(game_data)
        X = df['board'].apply(lambda x: [1 if v == "X" else -1 if v == "O" else 0 for v in x]).tolist()
        y = df['move']

        # dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # 1. Acurácia normal
        clf_normal = GradientBoostingClassifier()
        clf_normal.fit(X_train, y_train)
        y_pred_normal = clf_normal.predict(X_test)
        final_normal_accuracy = accuracy_score(y_test, y_pred_normal)

        # 2. Acurácia ajustada
        y_pred_adjusted = model.predict(X_test)
        final_adjusted_accuracy = accuracy_score(y_test, y_pred_adjusted)

        print(f"Acurácia normal final ao fechar a janela: {final_normal_accuracy * 100:.2f}%")
        print(f"Acurácia ajustada final ao fechar a janela: {final_adjusted_accuracy * 100:.2f}%")
    
    window.destroy()

#janela do jogo
def create_window():
    global buttons, window
    window = tk.Tk()
    window.title("Jogo da Velha (Automático)")

    for i in range(9):
        button = tk.Button(window, text=" ", font=("Arial", 24), width=5, height=2,
                           state=tk.DISABLED)
        button.grid(row=i // 3, column=i % 3)
        buttons.append(button)

    window.protocol("WM_DELETE_WINDOW", on_closing)

    # primeiro jogo
    window.after(game_speed, play_turn)

    window.mainloop()

if __name__ == "__main__":
    load_model()
    create_window()
