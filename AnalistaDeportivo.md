import requests
... import pandas as pd
... from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
... from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
... from sklearn.svm import SVC
... from sklearn.neural_network import MLPClassifier
... from sklearn.metrics import accuracy_score
... from sklearn.preprocessing import OneHotEncoder, StandardScaler
... from sklearn.pipeline import Pipeline
... import json
... 
... # Configura tus claves de API aquí
... API_KEY_FOOTBALL = 'your_api_football_key_here'  # Reemplaza con tu clave API válida
... 
... # Función para obtener datos de la API de fútbol
... def fetch_team_stats(team_id):
...     url = f'https://v3.football.api-sports.io/teams?id={team_id}'
...     headers = {
...         'x-apisports-key': API_KEY_FOOTBALL
...     }
...     response = requests.get(url, headers=headers)
...     
...     if response.status_code != 200:
...         print(f"Error fetching data: {response.status_code}, Response: {response.text}")
...         raise Exception(f"Error fetching data: {response.status_code}")
...     
...     try:
...         data = response.json()
...     except ValueError as e:
...         print("Error parsing JSON response:", e)
        print("Response text:", response.text)
        raise e
    
    return data['response'][0]

# Función para obtener datos adicionales del equipo
def fetch_team_info(team_id):
    url = f'https://v3.football.api-sports.io/teams/statistics?team={team_id}&season=2023'
    headers = {
        'x-apisports-key': API_KEY_FOOTBALL
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}, Response: {response.text}")
        raise Exception(f"Error fetching data: {response.status_code}")
    
    try:
        data = response.json()
    except ValueError as e:
        print("Error parsing JSON response:", e)
        print("Response text:", response.text)
        raise e
    
    return data['response']

# Función para obtener estadísticas de jugadores
def fetch_player_stats(team_id):
    url = f'https://v3.football.api-sports.io/players/squads?team={team_id}'
    headers = {
        'x-apisports-key': API_KEY_FOOTBALL
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}, Response: {response.text}")
        raise Exception(f"Error fetching data: {response.status_code}")
    
    try:
        data = response.json()
    except ValueError as e:
        print("Error parsing JSON response:", e)
        print("Response text:", response.text)
        raise e
    
    return data['response'][0]['players']

# Función para obtener historial de enfrentamientos
def fetch_head_to_head(team_id, opponent_id):
    url = f'https://v3.football.api-sports.io/fixtures/headtohead?h2h={team_id}-{opponent_id}'
    headers = {
        'x-apisports-key': API_KEY_FOOTBALL
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}, Response: {response.text}")
        raise Exception(f"Error fetching data: {response.status_code}")
    
    try:
        data = response.json()
    except ValueError as e:
        print("Error parsing JSON response:", e)
        print("Response text:", response.text)
        raise e
    
    return data['response']

# Función para obtener datos de clima (implementación ficticia)
def fetch_weather_data(city, date):
    # Aquí deberías implementar la llamada a la API del clima, este es un ejemplo ficticio.
    return {
        'avgtemp_c': 20,
        'condition': {'text': 'Clear'}
    }

# Función para obtener datos de lesiones
def fetch_injury_data(team_id):
    url = f'https://v3.football.api-sports.io/injuries?team={team_id}'
    headers = {
        'x-apisports-key': API_KEY_FOOTBALL
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}, Response: {response.text}")
        raise Exception(f"Error fetching data: {response.status_code}")
    
    try:
        data = response.json()
    except ValueError as e:
        print("Error parsing JSON response:", e)
        print("Response text:", response.text)
        raise e
    
    return data['response']

# Procesamiento de datos
def process_data(matches, team_info, player_stats, head_to_head, weather_data, injury_data):
    data = []
    for match in matches:
        if match['fixture']['status']['short'] != 'FT':
            continue
        
        home_team = match['teams']['home']['name']
        away_team = match['teams']['away']['name']
        home_score = match['goals']['home']
        away_score = match['goals']['away']
        result = 1 if home_score > away_score else 0 if home_score < away_score else 2
        
        # Características adicionales
        home_team_position = team_info['standings'][0][0]['rank'] if team_info['team']['name'] == home_team else 0
        away_team_position = team_info['standings'][0][0]['rank'] if team_info['team']['name'] == away_team else 0
        home_team_form = sum(m['goals']['home'] for m in matches[-5:]) / 5
        away_team_form = sum(m['goals']['away'] for m in matches[-5:]) / 5
        
        # Estadísticas de jugadores clave
        home_goals = sum(p['statistics'][0]['goals']['total'] for p in player_stats if p['team']['name'] == home_team)
        away_goals = sum(p['statistics'][0]['goals']['total'] for p in player_stats if p['team']['name'] == away_team)
        
        # Historial de enfrentamientos
        head_to_head_wins = sum(1 for h in head_to_head if h['teams']['home']['name'] == home_team and h['teams']['home']['winner'])
        head_to_head_losses = sum(1 for h in head_to_head if h['teams']['away']['name'] == away_team and h['teams']['away']['winner'])
        
        # Datos de clima
        weather_temp = weather_data['avgtemp_c']
        weather_condition = weather_data['condition']['text']
        
        # Datos de lesiones
        injuries = sum(1 for injury in injury_data if injury['team']['name'] == home_team or injury['team']['name'] == away_team)
        
        data.append({
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'result': result,
            'home_team_position': home_team_position,
            'away_team_position': away_team_position,
            'home_team_form': home_team_form,
            'away_team_form': away_team_form,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'head_to_head_wins': head_to_head_wins,
            'head_to_head_losses': head_to_head_losses,
            'weather_temp': weather_temp,
            'weather_condition': weather_condition,
            'injuries': injuries
        })
    
    df = pd.DataFrame(data)
    return df

# Análisis de datos
def analyze_data(df):
    print(df.describe())
    # Realiza análisis exploratorio adicional, como gráficos de correlación
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()

# Preprocesamiento para manejo de características categóricas
def preprocess_data(df):
    categorical_features = ['weather_condition']
    encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    df = df.drop(categorical_features, axis=1)
    df = pd.concat([df, encoded_df], axis=1)
    return df

# Modelado predictivo y ajuste de hiperparámetros
def train_model(df):
    df = preprocess_data(df)
    X = df.drop('result', axis=1)
    y = df['result']
    
    # Usar Pipeline para combinar escalado y modelado
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    scores = cross_val_score(best_model, X, y, cv=5)
    print(f"Cross-validated accuracy: {scores.mean()} (+/- {scores.std()})")
    print(f"Best Parameters: {grid_search.best_params_}")
    return best_model

def compare_models(df):
    df = preprocess_data(df)
    X = df.drop('result', axis=1)
    y = df['result']
    
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'NeuralNetwork': MLPClassifier(random_state=42)
    }
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        scores = cross_val_score(pipeline, X, y, cv=5)
        print(f"{name} accuracy: {scores.mean()} (+/- {scores.std()})")

# Ejemplo de uso
team_id = 33  # ID del equipo (por ejemplo, Manchester United)
opponent_id = 40  # ID del equipo oponente (por ejemplo, Liverpool FC)
city = 'Manchester'  # Ciudad del partido
date = '2023-04-24'  # Fecha del partido

try:
    matches = fetch_team_stats(team_id)
    team_info = fetch_team_info(team_id)
    player_stats = fetch_player_stats(team_id)
    head_to_head = fetch_head_to_head(team_id, opponent_id)
    weather_data = fetch_weather_data(city, date)
    injury_data = fetch_injury_data(team_id)

    processed_data = process_data(matches, team_info, player_stats, head_to_head, weather_data, injury_data)
    analyze_data(processed_data)

    # Entrenar el modelo
    model = train_model(processed_data)

    # Comparar modelos
    compare_models(processed_data)

except Exception as e:
    print(e)
