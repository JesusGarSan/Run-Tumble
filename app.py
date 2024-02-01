import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='centered')

st.title('Run and Tumble')
st.subheader('Movilidad y dinámica celular curso 2023-2024')
st.markdown('Jesús García Sánchez')
st.divider()
# ------------------------



def run_and_tumble(num_steps, step_size, tumble_prob, alpha):
    # Inicializamos la posición y la orientación inicial
    positions = np.zeros((num_steps + 1, 2))
    positions[0] = np.array([0.0, 0.0])
    orientation = np.random.rand() * 2 * np.pi  # Orientación inicial


    for i in range(1, num_steps + 1):
        
        # Movimiento "run"
        c_old = concentration(positions[i - 1])
        positions[i] = positions[i - 1] + step_size * np.array([np.cos(orientation), np.sin(orientation)])
        c_new = concentration(positions[i])
        
        if c_new > c_old:
            positions[i] += alpha * np.array([np.cos(orientation), np.sin(orientation)])
        
        # Tumble con probabilidad tumble_prob
        if np.random.rand() < tumble_prob:
            # La desviación en el tumble se obtiene de una distribución de probabilidad
            deviation = np.random.uniform(-np.pi, np.pi)
            orientation += deviation

    return positions

grad_x, grad_y = 0, 0

def concentration(position, grad_x = grad_x, grad_y = grad_y):
    return grad_x * position[0] + grad_y * position[1]


st.header('Trayectoria Run & Tumble sin quimiotaxis')

# Formulario para especificar los parámetros
cols = st.columns(2)

num_steps = cols[0].slider('Número de pasos:', min_value=50, max_value=500, value=200)
step_size = cols[1].slider('Tamaño del paso del Run:', min_value=0.05, max_value=1.0, value=0.1)
tumble_prob = cols[0].slider('Probabilidad de Tumble:', min_value=0.0, max_value=1.0, value=0.1)
#alfa = st.slider('Valor de alfa:', min_value=0.0, max_value=1.0, value=0.5)
n_simulations = st.slider('Número de simulaciones:', min_value=1, max_value=100, value=1)
alfa = 0

# Botón "Run" para ejecutar la simulación
if st.button('Simular'):

    trajectories = []
    fig, ax = plt.subplots()
    for _ in range(n_simulations):
        trajectory = run_and_tumble(num_steps, step_size, tumble_prob, alfa)
        trajectories.append(trajectory)

        # Visualización de la trayectoria
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color='red')#, label='Inicio')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='green', alpha = 0.5)#, label='Fin')

    


    # Visualización de la trayectoria   
    if n_simulations == 1:
        ax.plot(trajectory[:, 0], trajectory[:, 1], label='Trayectoria')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color='red', label='Inicio')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='green', alpha = 0.5, label='Fin')
    ax.set_title('Simulación de movimiento Run and Tumble')
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.legend()
    st.pyplot(fig)

# ------------------------------------------------------

st.divider()
st.header('Trayectoria Run & Tumble con quimiotaxis')

cols = st.columns(2)
grad_x = cols[0].number_input('Gradiente en x', value = 1.0, step = 0.1)
grad_y = cols[1].number_input('Gradiente en y', value = 0.0, step = 0.1)

def concentration(position, grad_x = grad_x, grad_y = grad_y):
    return grad_x * position[0] + grad_y * position[1]


# Formulario para especificar los parámetros
cols = st.columns(2)

col1, col2 = cols[1].columns(2)
alfa = st.slider('Valor de acción de la bacteria:', min_value=-1.0, max_value=1.0, value=0.5, help='Indica lo fuertemente que reacciona a difencias de gradiente. Valores positivos representan quimioatractantes y valores negativos representan quimirepulsores')
#alfa = 0

# Botón "Run" para ejecutar la simulación
if st.button('Simular', key = 2):

    trajectories = []
    fig, ax = plt.subplots()
    for _ in range(n_simulations):
        trajectory = run_and_tumble(num_steps, step_size, tumble_prob, alfa)
        trajectories.append(trajectory)

        # Visualización de la trayectoria
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color='red')#, label='Inicio')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='green', alpha = 0.5)#, label='Fin')

    


    # Visualización de la trayectoria   
    if n_simulations == 1:
        ax.plot(trajectory[:, 0], trajectory[:, 1], label='Trayectoria')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color='red', label='Inicio')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='green', alpha = 0.5, label='Fin')
    ax.set_title('Simulación de movimiento Run and Tumble')
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.legend()
    st.pyplot(fig)




# ------------------------------------------------------

st.divider()
st.header('Revisión del código')


st.code("""
def run_and_tumble(num_steps, step_size, tumble_prob, alpha):
    # Inicializamos la posición y la orientación inicial
    positions = np.zeros((num_steps + 1, 2))
    positions[0] = np.array([0.0, 0.0])
    orientation = np.random.rand() * 2 * np.pi  # Orientación inicial


    for i in range(1, num_steps + 1):
        
        # Movimiento "run"
        c_old = concentration(positions[i - 1])
        positions[i] = positions[i - 1] + step_size * np.array([np.cos(orientation), np.sin(orientation)])
        c_new = concentration(positions[i])
        
        if c_new > c_old:
            positions[i] += alpha * np.array([np.cos(orientation), np.sin(orientation)])
        
        # Tumble con probabilidad tumble_prob
        if np.random.rand() < tumble_prob:
            # La desviación en el tumble se obtiene de una distribución de probabilidad
            deviation = np.random.uniform(-np.pi, np.pi)
            orientation += deviation

    return positions

grad_x, grad_y = 0, 0

def concentration(position, grad_x = grad_x, grad_y = grad_y):
    return grad_x * position[0] + grad_y * position[1]
        """)
