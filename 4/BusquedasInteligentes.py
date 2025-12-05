import heapq
import random
from collections import deque

# -----------------------------
# Configuración del Puzzle
# -----------------------------

OBJETIVO = (1, 2, 3, 4, 5, 6, 7, 8, 0)
MOVS = {'arriba': -3, 'abajo': 3, 'izq': -1, 'der': 1}
COORD = {i: (i // 3, i % 3) for i in range(9)}  # coordenadas (fila, columna)

# -----------------------------
# Funciones base del 8-Puzzle
# -----------------------------

def es_resoluble(estado):
    inversions = 0
    lista = [n for n in estado if n != 0]
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if lista[i] > lista[j]:
                inversions += 1
    return inversions % 2 == 0


def generar_estado_aleatorio():
    while True:
        estado = tuple(random.sample(range(9), 9))
        if es_resoluble(estado):
            return estado


def funcion_sucesor(estado):
    sucesores = []
    i = estado.index(0)

    for mov, delta in MOVS.items():
        j = i + delta

        if mov == 'arriba' and i >= 3:
            pass
        elif mov == 'abajo' and i < 6:
            pass
        elif mov == 'izq' and i % 3 != 0:
            pass
        elif mov == 'der' and i % 3 != 2:
            pass
        else:
            continue

        nuevo = list(estado)
        nuevo[i], nuevo[j] = nuevo[j], nuevo[i]
        sucesores.append(tuple(nuevo))

    return sucesores


# -----------------------------
# Reconstrucción de camino
# -----------------------------

def reconstruir_camino(parent):
    camino = []
    cur = OBJETIVO
    while cur is not None:
        camino.append(cur)
        cur = parent[cur]
    return list(reversed(camino))


# -----------------------------------------
# BFS (Búsqueda Primero en Anchura)
# -----------------------------------------

def bfs_puzzle(inicial):
    if not es_resoluble(inicial):
        return None

    cola = deque([inicial])
    visitados = set([inicial])
    parent = {inicial: None}

    while cola:
        estado = cola.popleft()

        if estado == OBJETIVO:
            return reconstruir_camino(parent)

        for sucesor in funcion_sucesor(estado):
            if sucesor not in visitados:
                visitados.add(sucesor)
                parent[sucesor] = estado
                cola.append(sucesor)

    return None


# -----------------------------------------
# DFS limitada (para evitar loops)
# -----------------------------------------

def dfs_puzzle(inicial, limite=50):
    if not es_resoluble(inicial):
        return None

    stack = [(inicial, 0)]
    visitados = set([inicial])
    parent = {inicial: None}

    while stack:
        estado, profundidad = stack.pop()

        if estado == OBJETIVO:
            return reconstruir_camino(parent)

        if profundidad >= limite:
            continue

        for sucesor in funcion_sucesor(estado):
            if sucesor not in visitados:
                visitados.add(sucesor)
                parent[sucesor] = estado
                stack.append((sucesor, profundidad + 1))

    return None


# -----------------------------------------
# UCS (Dijkstra)
# -----------------------------------------

def ucs_puzzle(inicial):
    if not es_resoluble(inicial):
        return None

    INF = float('inf')
    dist = {inicial: 0}
    parent = {inicial: None}
    heap = [(0, inicial)]
    visitados = set()

    while heap:
        costo, estado = heapq.heappop(heap)

        if estado in visitados:
            continue
        visitados.add(estado)

        if estado == OBJETIVO:
            return reconstruir_camino(parent)

        for sucesor in funcion_sucesor(estado):
            nuevo_costo = costo + 1
            if nuevo_costo < dist.get(sucesor, INF):
                dist[sucesor] = nuevo_costo
                parent[sucesor] = estado
                heapq.heappush(heap, (nuevo_costo, sucesor))

    return None


# -----------------------------------------
# Heurística propia 
# -----------------------------------------

def heuristica_propia(estado):
    h = 0

    for idx, valor in enumerate(estado):
        if valor == 0:
            continue

        x1, y1 = COORD[idx]
        x2, y2 = COORD[valor]

        # Distancia Manhattan
        h += abs(x1 - x2) + abs(y1 - y2)

        # Conflictos lineales (misma fila)
        if x1 == x2:
            for j in range(9):
                if j != idx and COORD[j][0] == x1:
                    if estado[j] != 0 and COORD[estado[j]][0] == x1:
                        if (valor < estado[j] and j < idx) or (valor > estado[j] and j > idx):
                            h += 2

        # Conflictos lineales (misma columna)
        if y1 == y2:
            for j in range(9):
                if j != idx and COORD[j][1] == y1:
                    if estado[j] != 0 and COORD[estado[j]][1] == y1:
                        if (valor < estado[j] and j < idx) or (valor > estado[j] and j > idx):
                            h += 2

    return h


# -----------------------------------------
# A* (A estrella)
# -----------------------------------------

def a_estrella_puzzle(inicial):
    if not es_resoluble(inicial):
        return None

    heap = [(heuristica_propia(inicial), 0, inicial)]
    parent = {inicial: None}
    g_cost = {inicial: 0}
    visitados = set()

    while heap:
        f, g, estado = heapq.heappop(heap)

        if estado in visitados:
            continue
        visitados.add(estado)

        if estado == OBJETIVO:
            return reconstruir_camino(parent)

        for sucesor in funcion_sucesor(estado):
            nuevo_g = g + 1

            if sucesor not in g_cost or nuevo_g < g_cost[sucesor]:
                g_cost[sucesor] = nuevo_g
                parent[sucesor] = estado
                f_nuevo = nuevo_g + heuristica_propia(sucesor)
                heapq.heappush(heap, (f_nuevo, nuevo_g, sucesor))

    return None


# -----------------------------------------
# MAIN
# -----------------------------------------

if __name__ == "__main__":

    estado_inicial = generar_estado_aleatorio()

    print("Estado inicial:")
    for i in range(0, 9, 3):
        print(estado_inicial[i:i+3])

    print("\nSelecciona el algoritmo de búsqueda:")
    print("1. BFS (Anchura)")
    print("2. DFS (Profundidad limitada)")
    print("3. UCS (Costo uniforme / Dijkstra)")
    print("4. A* (A estrella con heurística propia)")

    opcion = input("\nIngresa el número del algoritmo: ")

    if opcion == "1":
        print("\n--- Ejecutando BFS ---")
        solucion = bfs_puzzle(estado_inicial)

    elif opcion == "2":
        print("\n--- Ejecutando DFS ---")
        solucion = dfs_puzzle(estado_inicial, limite=40)  # puedes ajustar límite

    elif opcion == "3":
        print("\n--- Ejecutando UCS ---")
        solucion = ucs_puzzle(estado_inicial)

    elif opcion == "4":
        print("\n--- Ejecutando A* ---")
        solucion = a_estrella_puzzle(estado_inicial)

    else:
        print("Opción no válida.")
        exit()

    if solucion:
        print(f"\nPasos a la solución: {len(solucion)-1}")
        for idx, paso in enumerate(solucion):
            print(f"\nPaso {idx}:")
            for i in range(0, 9, 3):
                print(paso[i:i+3])
    else:
        print("\nNo se encontró solución.")
