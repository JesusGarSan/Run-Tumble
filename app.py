import streamlit as st

import matplotlib.pyplot as plt




st.set_page_config(layout="wide")


def reiniciar():
    st.session_state.X=[]
    st.session_state.O=[]

def X(x,y,z):
        st.session_state.X.append([x,y,z])
def O(x,y,z):
    st.session_state.O.append([x,y,z])



if 'turno' not in st.session_state: st.session_state.turno='X'
if 'X' not in st.session_state: st.session_state.X=[]
if 'O' not in st.session_state: st.session_state.O=[]

COL1, COL2, COL3 = st.columns(3)
COL1.markdown('#    X')
COL3.markdown('#    O')
if COL2.button('Reiniciar', use_container_width=True): reiniciar()

with COL1.form('submit_x'):
    col1, col2, col3 = COL1.columns(3)
    x=col1.number_input('X', min_value=1, max_value=3, step=1)
    y=col2.number_input('Y', min_value=1, max_value=3, step=1)
    z=col3.number_input('Z', min_value=1, max_value=3, step=1)


    if st.session_state.turno=='X': disable_X=False
    else: disable_X=True
    if st.form_submit_button('Marcar', use_container_width=True, disabled=disable_X):
        if [x,y,z] not in st.session_state.X and [x,y,z] not in st.session_state.O:
            X(x,y,z)
            st.session_state.turno='O'
            st.experimental_rerun()

with COL3.form('submit_O'):
    col1, col2, col3 = COL3.columns(3)
    x=col1.number_input('X', min_value=1, max_value=3, step=1, key=1)
    y=col2.number_input('Y', min_value=1, max_value=3, step=1, key=2)
    z=col3.number_input('Z', min_value=1, max_value=3, step=1, key=3)

    if st.session_state.turno=='O': disable_O=False
    else: disable_O=True
    if st.form_submit_button('Marcar', use_container_width=True, disabled=disable_O):
        if [x,y,z] not in st.session_state.X and [x,y,z] not in st.session_state.O:
            O(x,y,z)
            st.session_state.turno='X'
            st.experimental_rerun()



x=st.session_state.X
o=st.session_state.O



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.grid(True)



for x in st.session_state.X:
    ax.scatter(x[0],x[1],x[2], c='red', marker='x', s=500)
for o in st.session_state.O:
    ax.scatter(o[0],o[1],o[2], c='blue', marker='8', s=500)

ax.set_xlim(0.5,3.5)
ax.set_ylim(0.5,3.5)
ax.set_zlim(0.5,3.5)

ax.set_xticks([1,2,3])
ax.set_yticks([1,2,3])
ax.set_zticks([1,2,3])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')



COL2.pyplot(fig)