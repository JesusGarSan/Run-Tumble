import streamlit as st
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

st.set_page_config(layout='centered')

st.title('Run and Tumble')
st.subheader('Movilidad y dinámica celular curso 2023-2024')
st.markdown('Jesús García Sánchez')
st.divider()
# ------------------------

global Cx, Cy
Cx, Cy = 0.0, 0.0
def concentration(x, y):
    return Cx * x + Cy * y # Distribución uniforme 

# Definción de la distribución de probabildiad de los ángulos de desviaicón de los tumbles
global tumble_gaussian
tumble_gaussian = truncnorm.rvs(-np.pi, np.pi, loc=0, scale=.25, size=1000)

# Definición de la función para producir un run and tumble
def run_and_tumble(num_steps, step_size, tumble_prob, alpha):
    # Inicializamos la posición y la orientación inicial
    positions = np.zeros((num_steps + 1, 2))
    positions[0] = np.array([0.0, 0.0])
    orientation = np.random.rand() * 2 * np.pi  # Orientación inicial

    for i in np.arange(1, num_steps + 1):
        c_old = concentration(positions[i - 1][0], positions[i - 1][1])
        
        # Movimiento "run"
        positions[i] = positions[i - 1] + step_size * np.array([np.cos(orientation), np.sin(orientation)])
        c_new = concentration(positions[i][0], positions[i][1])
       
        if c_new > c_old:
            positions[i] += alpha * np.array([np.cos(orientation), np.sin(orientation)])
        else:
            positions[i] -= alpha * np.array([np.cos(orientation), np.sin(orientation)])
        
        # Tumble con probabilidad tumble_prob
        if np.random.rand() < tumble_prob:
            # La desviación en el tumble se obtiene de una distribución de probabilidad
            deviation = tumble_gaussian[int(np.random.uniform(0,1000))]
            orientation += deviation

    return positions




# ----------------------------------------------------------------------------------------------------------

st.header('Trayectoria Run & Tumble sin quimiotaxis')

# Formulario para especificar los parámetros
cols = st.columns(2)

num_steps = cols[0].slider('Número de pasos:', min_value=50, max_value=500, value=200)
step_size = cols[1].slider('Tamaño del paso del Run:', min_value=0.05, max_value=1.0, value=0.1)
tumble_prob = cols[0].slider('Probabilidad de Tumble:', min_value=0.0, max_value=1.0, value=0.1)
#alfa = st.slider('Valor de alfa:', min_value=0.0, max_value=1.0, value=0.5)
scale = cols[1].slider('Desviación estándar de la gaussiana de tumble', min_value = 0.1, max_value = 1., value=.25)
n_simulations = st.slider('Número de simulaciones:', min_value=1, max_value=100, value=1)
alfa = 0
tumble_gaussian = truncnorm.rvs(-np.pi, np.pi, loc=0, scale=scale, size=1000)

# Botón "Run" para ejecutar la simulación
if st.button('Simular'):

    trajectories = []
    fig, ax = plt.subplots()
    for _ in range(n_simulations):
        trajectory = run_and_tumble(num_steps, step_size, tumble_prob, alfa)
        trajectories.append(trajectory)

        # Visualización de la trayectoria
        # ax.plot(trajectory[:, 0], trajectory[:, 1], c='blue')#, label='Trayectoria')
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

custom_concentration = st.checkbox('Introducir fórmula personalizada')
if custom_concentration:
    global code
    code = ''
    code = st.text_input('Introduce tu expresión deseada de C(x, y) (código Python)', value ='')
    if code != '':
        def concentration(x, y):
            return eval(code)
else:
    cols = st.columns(2)
    Cx = cols[0].number_input('Gradiente en x', value = 1.0, step = 0.1)
    Cy = cols[1].number_input('Gradiente en y', value = 0.0, step = 0.1)

    def concentration(x, y):
        return Cx * x + Cy * y # Distribución uniforme 


# Formulario para especificar los parámetros
cols = st.columns(2)

col1, col2 = cols[1].columns(2)
alfa = st.slider('Sensibildiad quimiotáxica de la bacteria:', min_value=-step_size, max_value=step_size, value=0.075, help='Indica lo fuertemente que reacciona a difencias de gradiente. Valores positivos representan quimioatractantes y valores negativos representan quimirepulsores')
#alfa = 0

histogram_check = cols[1].checkbox('Graficar distribución de resultados')
if histogram_check:
    n_bins = cols[1].number_input('Número de bins', min_value=1, value = 30)
# Botón "Run" para ejecutar la simulación
if cols[0].button('Simular', key = 2):

    trajectories = []
    fig, ax = plt.subplots()
    final_positions = np.zeros([n_simulations, 2])

    for i in range(n_simulations):
        trajectory = run_and_tumble(num_steps, step_size, tumble_prob, alfa)
        #trajectories.append(trajectory)
    
        final_positions[i] = trajectory[-1]

        # Visualización de la trayectoria
        # ax.plot(trajectory[:, 0], trajectory[:, 1], c='blue')#, label='Trayectoria')
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


    if histogram_check:
        x , bins_x = np.histogram(final_positions[:,0], bins =n_bins)
        y , bins_y = np.histogram(final_positions[:,1], bins =n_bins)

        bins_x = bins_x[:-1] + np.diff(bins_x) / 2
        bins_y = bins_y[:-1] + np.diff(bins_y) / 2

        xlim = np.max(np.abs([bins_x, bins_y]))

        fig, ax = plt.subplots(1,2, figsize=(10, 6))
        fig.suptitle(f"Número de simulaciones: {n_simulations}")
        ax[0].hist(final_positions[:,0], label='Posición final x', bins = n_bins)
        ax[0].vlines([np.ma.average(bins_x, weights=x)], 0, max(x), label = f"{np.ma.average(bins_x, weights=x)}", color='red')
        ax[0].set_xlim(xmin=-xlim, xmax=xlim)
        ax[0].legend()
        ax[1].hist(final_positions[:,1], label='Posición final y', bins = n_bins , color='orange')
        ax[1].vlines([np.ma.average(bins_y, weights=y)], 0, max(y), label = f"{np.ma.average(bins_y, weights=y)}", color='red')
        ax[1].set_xlim(xmin=-xlim, xmax=xlim)
        ax[1].legend()
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
